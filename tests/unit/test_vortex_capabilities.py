"""Unit tests for full Vortex GPGPU capabilities:
- DXA Async Copy Engine (TMA-style GMEM->LMEM, multicast, K-major transpose)
- TCU Tensor Core Unit (WGMMA/WMMA, 2:4 structured sparsity, MX block scaling, FEDP backends, lockstep)
- Bank conflict modeling and scratchpad arbitration
- Custom RISC-V encodings, hardware configurability, and performance CSRs
- Schema validation and AsyncOp IR integration
"""

import unittest

import numpy as np

from tritonflow.emit.ir import AsyncOp, Instr, Program
from tritonflow.emu.dxa import DxaDescriptor, DxaEmulator
from tritonflow.emu.hardware import BankConflictModel
from tritonflow.emu.tcu import TcuEmulator
from tritonflow.isa.schema import load_builtin, validate_schema


class TestVortexSchemaCapabilities(unittest.TestCase):
    """Verify that vortex_rvgpu.yaml declares all real Vortex hardware features."""

    def setUp(self) -> None:
        self.schema = load_builtin("vortex_rvgpu")

    def test_schema_loads_and_validates(self) -> None:
        self.assertEqual(self.schema.name, "vortex_rvgpu")
        self.assertEqual(self.schema.schema_version, 1)
        violations = validate_schema(self.schema)
        self.assertEqual(violations, [], f"Schema validation violations: {violations}")

    def test_hardware_config_mirroring_vx_config(self) -> None:
        cfg = self.schema.config
        self.assertIn("tcu", cfg)
        self.assertIn("dxa", cfg)
        self.assertIn("pipeline", cfg)
        self.assertIn("memory", cfg)

        self.assertEqual(cfg["tcu"]["tcu_type"], "TFR")
        self.assertTrue(cfg["tcu"]["sparse_enable"])
        self.assertTrue(cfg["tcu"]["mx_enable"])
        self.assertTrue(cfg["tcu"]["wgmma_enable"])

        self.assertEqual(cfg["dxa"]["max_inflight"], 8)
        self.assertTrue(cfg["dxa"]["multicast_enable"])
        self.assertTrue(cfg["dxa"]["kmajor_transpose_enable"])

    def test_performance_csr_registers(self) -> None:
        csrs = self.schema.csr_registers
        # DXA CSRs (0xB03-0xB07)
        self.assertEqual(csrs["DXA_TRANSFERS"], 0xB03)
        self.assertEqual(csrs["DXA_GMEM_READS"], 0xB04)
        self.assertEqual(csrs["DXA_GMEM_DEDUP"], 0xB05)
        self.assertEqual(csrs["DXA_LMEM_WRITES"], 0xB06)
        self.assertEqual(csrs["DXA_GMEM_LATENCY"], 0xB07)
        # TCU CSRs (upstream MPM class 11 multiplexed: 0xB03-0xB05)
        self.assertEqual(csrs["TCU_TBUF_STALLS"], 0xB03)
        self.assertEqual(csrs["TCU_TBUF_CACHE_HITS"], 0xB04)
        self.assertEqual(csrs["TCU_TBUF_HITS"], 0xB04)
        self.assertEqual(csrs["TCU_LMEM_READS"], 0xB05)
        self.assertEqual(csrs["CTA_CLUSTER_SIZE"], 0xCE0)

    def test_memory_spaces_with_banking(self) -> None:
        spaces = {s.name: s for s in self.schema.data_model.memory_spaces}
        self.assertIn("global", spaces)
        self.assertIn("scratch", spaces)
        self.assertIn("accum", spaces)
        self.assertIn("tcu_meta", spaces)

        scratch = spaces["scratch"]
        self.assertEqual(scratch.kind, "banked")
        self.assertEqual(scratch.banks, 4)
        self.assertEqual(scratch.interleave_bytes, 4)

    def test_dxa_instruction_definitions(self) -> None:
        dxa_1d = self.schema.instruction("DXA_COPY_1D")
        self.assertIsNotNone(dxa_1d)
        self.assertTrue(dxa_1d.is_async)
        self.assertEqual(dxa_1d.completion, "barrier_transaction")
        self.assertEqual(dxa_1d.encoding["op"], 9)

        dxa_2d = self.schema.instruction("DXA_COPY_2D")
        self.assertIsNotNone(dxa_2d)
        self.assertTrue(dxa_2d.is_async)

        dxa_km = self.schema.instruction("DXA_COPY_2D_KMAJOR")
        self.assertIsNotNone(dxa_km)
        self.assertTrue(dxa_km.is_async)

        dxa_mc = self.schema.instruction("DXA_COPY_MULTICAST")
        self.assertIsNotNone(dxa_mc)
        self.assertTrue(dxa_mc.is_async)

    def test_tcu_instruction_definitions(self) -> None:
        wmma16 = self.schema.instruction("TCU_WMMA16")
        self.assertIsNotNone(wmma16)
        self.assertEqual(wmma16.rule, "mac")
        self.assertEqual(wmma16.tile, {"m": 16, "n": 16})

        wgmma32 = self.schema.instruction("TCU_WGMMA32")
        self.assertIsNotNone(wgmma32)
        self.assertEqual(wgmma32.tile, {"m": 32, "n": 32})

        sp32 = self.schema.instruction("TCU_WGMMA_SP32")
        self.assertIsNotNone(sp32)
        self.assertTrue(sp32.sparse)
        self.assertEqual(sp32.compression_ratio, 2.0)
        self.assertEqual(sp32.encoding["op_type"], 4)

        mxfp8 = self.schema.instruction("TCU_WGMMA_MXFP8")
        self.assertIsNotNone(mxfp8)
        self.assertEqual(mxfp8.format, "mxfp8")
        self.assertEqual(mxfp8.block_scale_size, 32)

        # Aliases for backward compatibility
        self.assertIsNotNone(self.schema.instruction("TCU_MMA16"))
        self.assertIsNotNone(self.schema.instruction("TCU_MMA32"))


class TestBankConflictModel(unittest.TestCase):
    """Verify cycle-accurate 16-bank scratchpad arbitration and conflict model."""

    def setUp(self) -> None:
        self.model = BankConflictModel(num_banks=16, bank_width_bytes=4)

    def test_conflict_free_pattern(self) -> None:
        # Sequential 4-byte accesses mapping to banks 0, 1, 2, ..., 15
        addrs = [i * 4 for i in range(16)]
        stalls = self.model.analyze_access_pattern(addrs)
        self.assertEqual(stalls, 0)

    def test_all_conflicting_pattern(self) -> None:
        # 4 accesses all mapping to bank 0 (addresses 0, 64, 128, 192)
        # Bank stride = 16 banks * 4 bytes = 64 bytes
        addrs = [0, 64, 128, 192]
        stalls = self.model.analyze_access_pattern(addrs)
        self.assertEqual(stalls, 3)  # max_conflicts (4) - 1 = 3 stall cycles

    def test_two_way_conflicts(self) -> None:
        # Pairs of addresses mapping to banks 0, 1, 2
        addrs = [0, 64, 4, 68, 8, 72]
        stalls = self.model.analyze_access_pattern(addrs)
        self.assertEqual(stalls, 1)


class TestTcuEmulator(unittest.TestCase):
    """Verify Vortex TCU functional and timing emulation."""

    def test_fedp_backend_latencies(self) -> None:
        for backend, expected_lat in [("DPI", 4), ("DSP", 42), ("BHF", 13), ("TFR", 4)]:
            emu = TcuEmulator(fedp_type=backend)
            self.assertEqual(emu.fedp_latency, expected_lat)

    def test_lockstep_cta_conflict_gate(self) -> None:
        emu = TcuEmulator()
        self.assertFalse(emu.check_cta_conflict(0))
        self.assertTrue(emu.lock_cta(0))

        # CTA 1 should now be conflicted!
        self.assertTrue(emu.check_cta_conflict(1))
        self.assertFalse(emu.lock_cta(1))
        self.assertEqual(emu.perf.tbuf_stalls, 1)

        # Releasing CTA 0 frees the gate
        emu.unlock_cta(0)
        self.assertFalse(emu.check_cta_conflict(1))
        self.assertTrue(emu.lock_cta(1))

    def test_structured_sparsity_decompression(self) -> None:
        emu = TcuEmulator()
        # Compressed B matrix with shape (2, 4) -> original was (4, 4)
        b_compressed = np.array([[1.0, 2.0, 3.0, 4.0],
                                 [5.0, 6.0, 7.0, 8.0]], dtype=np.float32)
        # Default decompress expands to slots 0 and 2
        b_dense = emu.decompress_sparse(b_compressed)
        self.assertEqual(b_dense.shape, (4, 4))
        np.testing.assert_array_equal(b_dense[0], b_compressed[0])
        np.testing.assert_array_equal(b_dense[1], np.zeros(4))
        np.testing.assert_array_equal(b_dense[2], b_compressed[1])
        np.testing.assert_array_equal(b_dense[3], np.zeros(4))

    def test_microscaling_mx_format(self) -> None:
        emu = TcuEmulator()
        mat = np.ones((32, 32), dtype=np.float32)
        scales = np.array([2.0, 0.5], dtype=np.float32)
        scaled = emu.apply_mx_scale(mat, scale_factors=scales, block_size=32)
        self.assertEqual(scaled.shape, (32, 32))
        # First 32 elements scaled by 2.0
        np.testing.assert_array_equal(scaled.flatten()[:32], np.full(32, 2.0))

    def test_wgmma_execution(self) -> None:
        emu = TcuEmulator(fedp_type="TFR")
        a = np.ones((16, 16), dtype=np.float32)
        b = np.full((16, 16), 2.0, dtype=np.float32)
        c = np.zeros((16, 16), dtype=np.float32)

        result, cycles = emu.execute_wgmma(a, b, c, cta_id=0)
        # 16 * (1.0 * 2.0) = 32.0
        expected = np.full((16, 16), 32.0, dtype=np.float32)
        np.testing.assert_allclose(result, expected)
        self.assertGreater(cycles, 0)
        self.assertEqual(emu.perf.retired_ops, 1)


class TestDxaEmulator(unittest.TestCase):
    """Verify Vortex DXA async copy engine, multicast, and transaction barriers."""

    def test_descriptor_worklist_enumeration(self) -> None:
        dxa = DxaEmulator()
        desc = DxaDescriptor(base_addr=0, dim=2, elem_bytes=4, sizes=(4, 4), strides=(16, 4))
        work = dxa.enumerate_work_list(desc)
        self.assertEqual(len(work), 16)
        # Check first and second beat offsets
        self.assertEqual(work[0], (0, 0, False))
        self.assertEqual(work[1], (4, 4, False))

    def test_kmajor_transpose_scatter(self) -> None:
        dxa = DxaEmulator()
        desc = DxaDescriptor(base_addr=0, dim=2, elem_bytes=4, sizes=(4, 4), strides=(16, 4), dest_kmajor=True)
        work = dxa.enumerate_work_list(desc)
        # In K-major transpose: element (r=0, c=1) goes to SMEM (c*rows + r)*4 = 1*4*4 = 16
        g_off, s_off, _ = work[1]  # r=0, c=1
        self.assertEqual(g_off, 4)
        self.assertEqual(s_off, 16)

    def test_async_copy_multicast_and_barrier(self) -> None:
        dxa = DxaEmulator(gmem_latency=10)
        gmem_data = np.arange(16, dtype=np.float32)
        smem_target = np.zeros(32, dtype=np.float32)

        desc = DxaDescriptor(
            base_addr=0,
            dim=1,
            elem_bytes=4,
            sizes=(16,),
            smem_stride=16 * 4,  # stride between CTAs in multicast
        )

        # Multicast to 2 CTAs (cta_mask = 0b11 = 0x3)
        handle = dxa.issue_copy(
            desc,
            smem_base=0,
            gmem_base=0,
            barrier_id=2,
            cta_mask=0x3,
            gmem_data=gmem_data,
            smem_target=smem_target,
        )

        self.assertFalse(dxa.is_barrier_complete(2))
        # GMEM read deduplication verified
        self.assertEqual(dxa.perf.gmem_reads, 16)
        self.assertEqual(dxa.perf.gmem_dedup_savings, 16)  # 16 reads saved!
        self.assertEqual(dxa.perf.lmem_writes, 32)         # written to 2 CTAs

        # Advance clock to completion
        completed = dxa.tick(cycles=100)
        self.assertIn(handle, completed)
        self.assertTrue(dxa.is_barrier_complete(2))
        self.assertEqual(dxa.perf.transfers_completed, 1)

        # Verify multicast data landed in both CTA regions
        np.testing.assert_array_equal(smem_target[:16], gmem_data)
        np.testing.assert_array_equal(smem_target[16:32], gmem_data)


class TestAsyncOpIR(unittest.TestCase):
    """Verify AsyncOp integration into Program IR."""

    def test_async_op_in_program(self) -> None:
        from tritonflow.emit.ir import SourceRef
        instr_launch = Instr(name="DXA_COPY_2D", source_ops=(SourceRef("tt.load", 1, 1),), cost=10.0)
        instr_wait = Instr(name="BARRIER_WAIT", source_ops=(SourceRef("vx.bar", 2, 1),), cost=4.0)
        async_op = AsyncOp(
            handle="dxa_tx_1",
            launch_instr=instr_launch,
            wait_instr=instr_wait,
            completion_barrier=1,
        )

        prog = Program(
            isa_name="vortex_rvgpu",
            schema_version=1,
            kernel_name="test_vortex_async",
            instrs=(instr_launch, instr_wait),
            total_cost=14.0,
            async_ops=(async_op,),
        )

        self.assertEqual(len(prog.async_ops), 1)
        self.assertEqual(prog.async_ops[0].handle, "dxa_tx_1")
        self.assertEqual(prog.async_ops[0].completion_barrier, 1)


if __name__ == "__main__":
    unittest.main()
