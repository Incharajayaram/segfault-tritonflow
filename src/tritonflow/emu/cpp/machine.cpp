// machine.cpp — MachineState and instruction execution.
//
// Mirrors exec.py. The emulator is a small machine that consumes the
// emitted instruction stream and nothing else.

#include "machine.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <limits>
#include <functional>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace tritonflow::emu {

// ---- Helpers ------------------------------------------------------------ //

static const char* PROGRAM_ID_AXES[] = {"x", "y", "z"};

static bool is_memory_instruction(const std::string& name) {
    return name == "DMA1D" || name == "DMA2D" || name == "LDG" || name == "LDS2D" ||
           name == "STG" || name == "LDS" || name == "STS" || name == "BARRIER" ||
           name == "BARRIER_EXPECT_TX" || name == "BARRIER_WAIT" || name == "STG2D" ||
           name.rfind("DXA_", 0) == 0 || name == "TCU_LD_META";
}

static bool is_mac_instruction(const std::string& name) {
    return name == "MAC8" || name == "MAC16" || name == "OPU8" || name == "OPU32" ||
           name.rfind("TCU_MMA", 0) == 0 || name.rfind("TCU_WGMMA", 0) == 0 ||
           name == "TCU_WMMA16";
}

static bool is_elementwise_instruction(const std::string& name) {
    if (name.rfind("EPI_", 0) == 0 || name.rfind("VPU_", 0) == 0) return true;
    return name == "EPI" || name == "VPU" || name == "CLAMP" ||
           name == "VADD" || name == "VMUL" || name == "VDIV" ||
           name == "VSUB" || name == "VMOD" || name == "VRELU" ||
           name == "VCLAMP" || name == "VNEG" || name == "VABS" ||
           name == "VMAX" || name == "VMIN" ||
           name == "LI" || name == "MOVI" ||
           name == "VEXPAND_DIMS" || name == "VBROADCAST" || name == "VSPLAT" ||
           name == "VMAKE_RANGE" || name == "VPID" || name == "VCMP" ||
           name == "VCVT";
}

/// Parse the k=v;... descriptor key into its fields.
static std::map<std::string, std::string> descriptor_fields(
    const std::optional<std::string>& key
) {
    std::map<std::string, std::string> fields;
    if (!key || key->empty()) return fields;

    std::istringstream ss(*key);
    std::string part;
    while (std::getline(ss, part, ';')) {
        auto eq = part.find('=');
        if (eq != std::string::npos) {
            std::string name = part.substr(0, eq);
            std::string value = part.substr(eq + 1);
            // Trim whitespace.
            auto ltrim = [](std::string& s) {
                s.erase(0, s.find_first_not_of(" \t"));
            };
            auto rtrim = [](std::string& s) {
                auto pos = s.find_last_not_of(" \t");
                if (pos != std::string::npos) s.erase(pos + 1);
            };
            ltrim(name); rtrim(name);
            ltrim(value); rtrim(value);
            fields[name] = value;
        }
    }
    return fields;
}

/// Parse sizes from a descriptor key like "sizes=[64, 64]".
static std::vector<int> parse_sizes(const std::optional<std::string>& key) {
    auto fields = descriptor_fields(key);
    auto it = fields.find("sizes");
    if (it == fields.end()) return {};

    std::string raw = it->second;
    // Strip brackets.
    if (!raw.empty() && raw.front() == '[') raw.erase(0, 1);
    if (!raw.empty() && raw.back() == ']') raw.pop_back();

    std::vector<int> result;
    std::istringstream ss(raw);
    std::string item;
    while (std::getline(ss, item, ',')) {
        auto ltrim = [](std::string& s) {
            s.erase(0, s.find_first_not_of(" \t"));
        };
        ltrim(item);
        if (!item.empty()) {
            result.push_back(std::stoi(item));
        }
    }
    return result;
}

/// Get the declared shape from an instruction's constrained_on field.
static std::vector<int> declared_shape(const Instr& instr) {
    return parse_sizes(instr.constrained_on);
}

/// Find which inputs are pointer (buffer) inputs by scanning MemRef operands.
static std::set<std::string> pointer_inputs(const Program& program) {
    std::set<std::string> names;
    for (const auto* instr : program.instructions()) {
        for (const auto& role : instr->roles) {
            auto it = instr->operands.find(role);
            if (it == instr->operands.end()) continue;
            if (auto* memref = std::get_if<MemRef>(&it->second)) {
                auto fields = descriptor_fields(memref->access_key);
                auto base_it = fields.find("base");
                names.insert(base_it != fields.end() ? base_it->second : memref->base);
            }
        }
    }
    return names;
}

/// Require an operand by role, throw if missing.
static const Operand& require_operand(const Instr& instr, const std::string& role) {
    const Operand* op = instr.operand(role);
    if (!op) {
        throw UnsupportedInstruction(
            instr.name + " is missing required role '" + role + "'");
    }
    return *op;
}

/// Get the mask operand if present.
static const Value* resolve_mask(const Instr& instr, const MachineState& state) {
    const Operand* mask_op = instr.operand("mask");
    if (!mask_op) return nullptr;
    // We store it in a thread_local to avoid dangling pointer issues.
    thread_local Value mask_val;
    mask_val = state.resolve(*mask_op);
    return &mask_val;
}

/// Get the "in*" operands in role order.
static std::vector<Value> input_operands(const Instr& instr, const MachineState& state) {
    std::vector<Value> result;
    for (const auto& role : instr.roles) {
        if (role.substr(0, 2) == "in") {
            result.push_back(state.resolve(instr.operands.at(role)));
        }
    }
    return result;
}

/// Convert int64_t indices from a Value.
static std::vector<int64_t> to_int64_vec(const Value& v) {
    std::vector<int64_t> result(v.data.size());
    for (size_t i = 0; i < v.data.size(); ++i) {
        result[i] = static_cast<int64_t>(v.data[i]);
    }
    return result;
}

// ---- MachineState ------------------------------------------------------- //

MachineState MachineState::from_program(
    const Program& program,
    const std::map<std::string, Value>& inputs,
    const PrecisionPolicy& pol,
    int gx, int gy, int gz
) {
    MachineState state;
    state.program = &program;
    state.policy = pol;
    state.grid[0] = gx;
    state.grid[1] = gy;
    state.grid[2] = gz;

    auto ptrs = pointer_inputs(program);

    // Build storage: lay out buffers contiguously in flat memory.
    int offset = 0;
    std::vector<const Value*> buffer_order;
    for (const auto& name : program.inputs) {
        if (ptrs.find(name) == ptrs.end()) continue;
        auto it = inputs.find(name);
        if (it == inputs.end()) {
            throw MissingInput(
                "pointer input " + name + " was not supplied; the kernel addresses "
                "it and the emulator will not fabricate storage for it");
        }
        Storage s;
        s.name = name;
        s.base = offset;
        s.length = static_cast<int>(it->second.data.size());
        s.shape = it->second.shape;
        state.storages[name] = s;
        buffer_order.push_back(&it->second);
        offset += s.length;
    }

    // Concatenate all buffers into flat memory.
    state.memory.resize(offset);
    int pos = 0;
    for (const auto* buf : buffer_order) {
        std::memcpy(state.memory.data() + pos, buf->data.data(),
                     buf->data.size() * sizeof(float));
        pos += static_cast<int>(buf->data.size());
    }

    // Bind inputs.
    for (const auto& name : program.inputs) {
        auto st_it = state.storages.find(name);
        if (st_it != state.storages.end()) {
            state.values[name] = Value::from_int(st_it->second.base);
        } else {
            auto in_it = inputs.find(name);
            if (in_it != inputs.end()) {
                state.values[name] = in_it->second;
            } else {
                // Check program ID axes.
                bool found = false;
                for (int ax = 0; ax < 3; ++ax) {
                    if (name == PROGRAM_ID_AXES[ax]) {
                        state.values[name] = Value::from_int(state.grid[ax]);
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    throw MissingInput(
                        "input " + name + " was not supplied and is not a "
                        "program-id axis; the emulator refuses to default a "
                        "value the kernel reads");
                }
            }
        }
    }

    return state;
}

Value MachineState::resolve(const Operand& operand) const {
    if (auto* imm = std::get_if<Imm>(&operand)) {
        if (imm->is_int()) {
            return Value::from_int(imm->as_int());
        }
        return Value::from_float(imm->as_double());
    }
    if (auto* ssa = std::get_if<SsaRef>(&operand)) {
        auto it = values.find(ssa->name);
        if (it != values.end()) {
            return it->second;
        }
        throw MissingInput(
            ssa->name + " is read but nothing produced it; a re-threaded "
            "loop value with no producer is an unexecutable program, not a zero");
    }
    if (auto* mem = std::get_if<MemRef>(&operand)) {
        auto it = values.find(mem->base);
        if (it != values.end()) {
            if (!it->second.is_scalar()) {
                return it->second;
            }
            int64_t base_addr = it->second.scalar_i();
            if (!mem->access_key || mem->access_key->empty()) {
                auto sit = storages.find(mem->base);
                if (sit != storages.end()) {
                    std::vector<float> addrs(sit->second.length);
                    for (int i = 0; i < sit->second.length; ++i) {
                        addrs[i] = static_cast<float>(sit->second.base + i);
                    }
                    return Value::from_array(std::move(addrs), sit->second.shape, true);
                }
                return it->second;
            }
            // Parse descriptor fields
            auto fields = descriptor_fields(mem->access_key);
            auto git = fields.find("is_gather");
            if (git != fields.end() && git->second == "True") {
                auto idx_it = fields.find("indices");
                if (idx_it != fields.end()) {
                    auto vit = values.find(idx_it->second);
                    if (vit != values.end()) {
                        std::vector<float> addrs(vit->second.data.size());
                        for (size_t i = 0; i < vit->second.data.size(); ++i) {
                            addrs[i] = static_cast<float>(base_addr + static_cast<int64_t>(vit->second.data[i]));
                        }
                        return Value::from_array(std::move(addrs), vit->second.shape, true);
                    }
                }
            }
            auto sizes = parse_sizes(mem->access_key);
            if (sizes.empty()) {
                return Value::from_array({static_cast<float>(base_addr)}, {1}, true);
            }

            // Parse strides
            std::vector<int> strides;
            auto s_it = fields.find("strides");
            if (s_it != fields.end()) {
                std::string raw = s_it->second;
                if (!raw.empty() && raw.front() == '[') raw.erase(0, 1);
                if (!raw.empty() && raw.back() == ']') raw.pop_back();
                std::istringstream ss(raw);
                std::string item;
                while (std::getline(ss, item, ',')) {
                    item.erase(0, item.find_first_not_of(" 	"));
                    auto pos = item.find_last_not_of(" 	");
                    if (pos != std::string::npos) item.erase(pos + 1);
                    if (item.empty()) continue;
                    try {
                        strides.push_back(std::stoi(item));
                    } catch (...) {
                        auto vit = values.find(item);
                        if (vit != values.end()) {
                            strides.push_back(static_cast<int>(vit->second.scalar_i()));
                        } else {
                            strides.push_back(1);
                        }
                    }
                }
            }

            // Parse offsets
            std::vector<int64_t> offsets;
            auto o_it = fields.find("offsets");
            if (o_it != fields.end()) {
                std::string raw = o_it->second;
                if (!raw.empty() && raw.front() == '[') raw.erase(0, 1);
                if (!raw.empty() && raw.back() == ']') raw.pop_back();
                std::istringstream ss(raw);
                std::string item;
                while (std::getline(ss, item, ',')) {
                    item.erase(0, item.find_first_not_of(" 	"));
                    auto pos = item.find_last_not_of(" 	");
                    if (pos != std::string::npos) item.erase(pos + 1);
                    if (item.empty()) continue;
                    int64_t val = 0;
                    std::istringstream term_ss(item);
                    std::string term;
                    while (std::getline(term_ss, term, '+')) {
                        term.erase(0, term.find_first_not_of(" 	"));
                        auto p = term.find_last_not_of(" 	");
                        if (p != std::string::npos) term.erase(p + 1);
                        if (term.empty()) continue;
                        std::istringstream prod_ss(term);
                        std::string factor;
                        int64_t prod = 1;
                        while (std::getline(prod_ss, factor, '*')) {
                            factor.erase(0, factor.find_first_not_of(" 	"));
                            auto fp = factor.find_last_not_of(" 	");
                            if (fp != std::string::npos) factor.erase(fp + 1);
                            if (factor.empty()) continue;
                            try {
                                prod *= std::stoll(factor);
                            } catch (...) {
                                if (factor == "%pid" || factor == "%pid_m" || factor == "%pid_x" || factor == "pid") {
                                    prod *= grid[0];
                                } else if (factor == "%pid_n" || factor == "%pid_y" || factor == "pid_n") {
                                    prod *= grid[1];
                                } else if (factor == "%pid_k" || factor == "%pid_z" || factor == "pid_k") {
                                    prod *= grid[2];
                                } else {
                                    auto vit = values.find(factor);
                                    if (vit != values.end()) {
                                        prod *= vit->second.scalar_i();
                                    }
                                }
                            }
                        }
                        val += prod;
                    }
                    offsets.push_back(val);
                }
            }

            bool is_loop_carried = (fields.count("loop_carried") && fields["loop_carried"] == "True");
            int inc = 0;
            if (fields.count("increment")) {
                try {
                    inc = std::stoi(fields["increment"]);
                } catch (...) {
                    auto vit = values.find(fields["increment"]);
                    if (vit != values.end()) inc = static_cast<int>(vit->second.scalar_i());
                }
            }

            int stride_inc = 1;
            if (is_loop_carried && !strides.empty()) {
                if (strides.size() == 1) {
                    stride_inc = strides[0];
                } else if (strides.size() == 2) {
                    std::string raw_strides = fields.count("strides") ? fields["strides"] : "";
                    if (raw_strides.find("k") != std::string::npos || raw_strides.find("K") != std::string::npos) {
                        auto comma = raw_strides.find(',');
                        std::string s0 = raw_strides.substr(0, comma);
                        if (s0.find("k") != std::string::npos || s0.find("K") != std::string::npos) {
                            stride_inc = strides[0];
                        } else {
                            stride_inc = strides[1];
                        }
                    } else if (sizes.size() == 2 && sizes[0] == inc && sizes[1] != inc) {
                        stride_inc = strides[0];
                    } else if (sizes.size() == 2 && sizes[1] == inc && sizes[0] != inc) {
                        stride_inc = strides[1];
                    } else if (mem->base.rfind("%b", 0) == 0 || mem->base.rfind("b", 0) == 0) {
                        stride_inc = strides[0];
                    } else {
                        stride_inc = strides[1];
                    }
                }
            }

            int64_t loop_offset = is_loop_carried ? (static_cast<int64_t>(loop_iteration) * inc * stride_inc) : 0;

            if (sizes.size() == 1) {
                int n = sizes[0];
                int64_t off0 = offsets.empty() ? 0 : offsets[0];
                int64_t str0 = strides.empty() ? 1 : strides[0];
                std::vector<float> addrs(n);
                for (int i = 0; i < n; ++i) {
                    addrs[i] = static_cast<float>(base_addr + loop_offset + off0 + i * str0);
                }
                return Value::from_array(std::move(addrs), sizes, true);
            } else if (sizes.size() == 2) {
                int rows = sizes[0], cols = sizes[1];
                int64_t off0 = offsets.empty() ? 0 : offsets[0];
                int64_t off1 = offsets.size() < 2 ? 0 : offsets[1];
                int64_t str0 = strides.empty() ? cols : strides[0];
                int64_t str1 = strides.size() < 2 ? 1 : strides[1];
                std::vector<float> addrs(rows * cols);
                for (int r = 0; r < rows; ++r) {
                    for (int c = 0; c < cols; ++c) {
                        addrs[r * cols + c] = static_cast<float>(base_addr + loop_offset + off0 + off1 + r * str0 + c * str1);
                    }
                }
                return Value::from_array(std::move(addrs), sizes, true);
            }
            return Value::from_int(base_addr);
        }
        throw MissingInput(
            mem->base + " is read but nothing produced it");
    }
    throw std::runtime_error("cannot resolve operand");
}

void MachineState::bind(const std::string& name, Value value) {
    values[name] = std::move(value);
}

Value MachineState::gather(const Value& indices, const Value* mask) const {
    auto flat = to_int64_vec(indices);
    check_bounds(flat);

    std::vector<float> out(flat.size());
    for (size_t i = 0; i < flat.size(); ++i) {
        out[i] = memory[flat[i]];
    }

    if (mask) {
        for (size_t i = 0; i < flat.size() && i < mask->data.size(); ++i) {
            if (mask->data[i] == 0.0f) {
                out[i] = 0.0f;
            }
        }
    }
    return Value::from_array(std::move(out), indices.shape);
}

void MachineState::scatter(const Value& indices, const Value& vals, const Value* mask) {
    auto flat = to_int64_vec(indices);
    check_bounds(flat);

    for (size_t i = 0; i < flat.size(); ++i) {
        if (mask && i < mask->data.size() && mask->data[i] == 0.0f) {
            continue;
        }
        float v = (i < vals.data.size()) ? vals.data[i] : vals.data[0];
        memory[flat[i]] = v;
    }
    mark_written(flat);
}

void MachineState::check_bounds(const std::vector<int64_t>& flat) const {
    if (flat.empty()) return;
    int64_t lo = *std::min_element(flat.begin(), flat.end());
    int64_t hi = *std::max_element(flat.begin(), flat.end());
    if (lo < 0 || hi >= static_cast<int64_t>(memory.size())) {
        std::ostringstream oss;
        oss << "access out of emulated storage: index range ["
            << lo << ", " << hi << "] but memory has "
            << memory.size() << " elements; refusing to zero-fill";
        throw StorageError(oss.str());
    }
}

void MachineState::mark_written(const std::vector<int64_t>& flat) {
    if (flat.empty()) return;
    int64_t lo = *std::min_element(flat.begin(), flat.end());
    int64_t hi = *std::max_element(flat.begin(), flat.end());
    for (const auto& [name, storage] : storages) {
        if (lo < storage.end() && hi >= storage.base) {
            written.insert(name);
        }
    }
}

std::map<std::string, Value> MachineState::outputs() const {
    std::map<std::string, Value> out;
    for (const auto& name : written) {
        auto it = storages.find(name);
        if (it == storages.end()) continue;
        const Storage& s = it->second;
        std::vector<float> buf(memory.begin() + s.base,
                               memory.begin() + s.end());
        out[name] = Value::from_array(std::move(buf), s.shape);
    }
    return out;
}

// ---- Instruction dispatch ----------------------------------------------- //

// Forward declarations.
static void apply_memory(const Instr& instr, MachineState& state);
static void apply_mac(const Instr& instr, MachineState& state, const PrecisionPolicy& policy);
static void apply_elementwise(const Instr& instr, MachineState& state);

void apply(const Instr& instr, MachineState& state, const PrecisionPolicy& policy) {
    if (instr.name.rfind("BARRIER", 0) == 0) {
        return;
    }
    if (is_memory_instruction(instr.name)) {
        apply_memory(instr, state);
        return;
    }
    if (is_mac_instruction(instr.name)) {
        apply_mac(instr, state, policy);
        return;
    }
    if (is_elementwise_instruction(instr.name)) {
        apply_elementwise(instr, state);
        return;
    }
    throw UnsupportedInstruction(
        "instruction '" + instr.name + "' is not implemented by this machine; "
        "an unimplemented instruction is a refusal, not a no-op");
}

// ---- Memory instructions ------------------------------------------------ //

// A memory instruction may only move data through spaces its schema entry lists.
// Mirrors exec.py::_check_space so the two emulators refuse the same programs.
static void check_space(const Instr& instr, const MachineState& state, const MemRef& ref, bool is_src) {
    const auto& table = state.program->transfers;
    auto it = table.find(instr.name);
    if (it == table.end()) {
        throw UnsupportedInstruction(
            "address space violation: " + instr.name +
            " declares no transfers, so a '" + ref.space + "' access cannot be verified");
    }
    for (const auto& move : it->second) {
        if ((is_src ? move.first : move.second) == ref.space) {
            return;
        }
    }
    std::string listed;
    for (const auto& move : it->second) {
        listed += (listed.empty() ? "" : ", ") + move.first + ">" + move.second;
    }
    throw UnsupportedInstruction(
        "address space violation: " + instr.name + (is_src ? " reads '" : " writes '") + ref.space +
        "' but its schema transfers are [" + listed + "]");
}

static void apply_memory(const Instr& instr, MachineState& state) {
    const Value* mask = resolve_mask(instr, state);

    // Store: dst is a MemRef.
    const Operand* dst_op = instr.operand("dst");
    if (dst_op && std::holds_alternative<MemRef>(*dst_op)) {
        check_space(instr, state, std::get<MemRef>(*dst_op), false);
        Value indices = state.resolve(*dst_op);
        const Operand& val_op = require_operand(instr, "value");
        Value payload = state.resolve(val_op);
        state.scatter(indices, payload, mask);
        return;
    }

    // Load: src is a MemRef.
    const Operand& src_op = require_operand(instr, "src");
    if (!std::holds_alternative<MemRef>(src_op)) {
        throw UnsupportedInstruction(
            instr.name + " has neither a dst nor a src memory operand");
    }
    check_space(instr, state, std::get<MemRef>(src_op), true);
    if (instr.defs.empty()) {
        throw UnsupportedInstruction(instr.name + " load defines no value to bind");
    }
    Value src_val = state.resolve(src_op);
    Value loaded = state.gather(src_val, mask);

    // Reshape to declared shape if available.
    auto shape = declared_shape(instr);
    if (!shape.empty()) {
        loaded.shape = shape;
    }
    state.bind(instr.defs[0], std::move(loaded));
}

// ---- MAC instructions --------------------------------------------------- //

static void apply_mac(const Instr& instr, MachineState& state, const PrecisionPolicy& policy) {
    Value a = state.resolve(require_operand(instr, "a"));
    Value b = state.resolve(require_operand(instr, "b"));
    Value acc_val = state.resolve(require_operand(instr, "acc"));

    // Determine dimensions from shapes.
    int m = a.shape.size() >= 2 ? a.shape[0] : static_cast<int>(a.data.size());
    int k = a.shape.size() >= 2 ? a.shape[1] : 1;
    int n = b.shape.size() >= 2 ? b.shape[1] : static_cast<int>(b.data.size());

    // Prepare accumulator.
    std::vector<float> acc(m * n, 0.0f);
    if (!acc_val.is_scalar()) {
        for (size_t i = 0; i < std::min(acc.size(), acc_val.data.size()); ++i) {
            acc[i] = acc_val.data[i];
        }
    }

    // multiply_accumulate.
    policy.multiply_accumulate(acc.data(), a.data.data(), m, k, b.data.data(), n);

    if (instr.defs.empty()) {
        throw UnsupportedInstruction(instr.name + " defines no accumulator value");
    }
    state.bind(instr.defs[0], Value::from_array(std::move(acc), {m, n}));
}

// ---- Elementwise instructions ------------------------------------------- //

// Elementwise op handlers.
using ElemHandler = std::function<Value(const Instr&, MachineState&, const std::vector<int>&)>;

static Value elem_const(const Instr& instr, MachineState& state, const std::vector<int>& shape) {
    Value v = state.resolve(require_operand(instr, "value"));
    if (shape.empty()) return v;
    size_t n = 1;
    for (int d : shape) n *= d;
    std::vector<float> data(n, v.scalar_f());
    return Value::from_array(std::move(data), shape, v.is_int);
}

static Value elem_program_id(const Instr& instr, MachineState& state, const std::vector<int>&) {
    Value v = state.resolve(require_operand(instr, "value"));
    int axis = static_cast<int>(v.scalar_i());
    if (axis >= 3) {
        throw UnsupportedInstruction("program id axis out of range");
    }
    return Value::from_int(state.grid[axis]);
}

static Value elem_make_range(const Instr& instr, MachineState& state, const std::vector<int>& shape) {
    Value v = state.resolve(require_operand(instr, "value"));
    int length = static_cast<int>(v.scalar_i());
    std::vector<float> data(length);
    for (int i = 0; i < length; ++i) data[i] = static_cast<float>(i);
    return Value::from_array(std::move(data), {length}, true);
}

static Value elem_splat(const Instr& instr, MachineState& state, const std::vector<int>& shape) {
    auto ops = input_operands(instr, state);
    if (ops.empty()) throw UnsupportedInstruction("splat has no input");
    float val = ops[0].scalar_f();
    auto sh = shape.empty() ? std::vector<int>{1} : shape;
    size_t n = 1;
    for (int d : sh) n *= d;
    std::vector<float> data(n, val);
    return Value::from_array(std::move(data), sh, ops[0].is_int);
}

static Value elem_broadcast(const Instr& instr, MachineState& state, const std::vector<int>& shape) {
    auto ops = input_operands(instr, state);
    if (ops.empty()) throw UnsupportedInstruction("broadcast has no input");
    const Value& src = ops[0];
    size_t n = 1;
    for (int d : shape) n *= d;

    // Simple broadcast: repeat the source data to fill the target shape.
    std::vector<float> data(n);
    if (src.data.empty()) {
        std::fill(data.begin(), data.end(), 0.0f);
    } else {
        for (size_t i = 0; i < n; ++i) {
            data[i] = src.data[i % src.data.size()];
        }
    }
    return Value::from_array(std::move(data), shape, src.is_int);
}

static Value elem_expand_dims(const Instr& instr, MachineState& state, const std::vector<int>& shape) {
    auto ops = input_operands(instr, state);
    if (ops.empty()) throw UnsupportedInstruction("expand_dims has no input");
    if (shape.empty()) throw UnsupportedInstruction("expand_dims has no recorded result shape");
    size_t n = 1;
    for (int d : shape) n *= d;
    if (n != ops[0].data.size()) {
        throw UnsupportedInstruction(
            "expand_dims result shape cannot be that reshape");
    }
    return Value::from_array(
        std::vector<float>(ops[0].data), shape, ops[0].is_int);
}

// Binary ops.
static Value elem_addi(const Instr& instr, MachineState& state, const std::vector<int>& shape) {
    auto ops = input_operands(instr, state);
    size_t n = std::max(ops[0].data.size(), ops[1].data.size());
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) {
        int64_t a = static_cast<int64_t>(ops[0].data[i % ops[0].data.size()]);
        int64_t b = static_cast<int64_t>(ops[1].data[i % ops[1].data.size()]);
        out[i] = static_cast<float>(a + b);
    }
    auto sh = shape.empty() ? (ops[0].shape.empty() ? ops[1].shape : ops[0].shape) : shape;
    return Value::from_array(std::move(out), sh, true);
}

static Value elem_addf(const Instr& instr, MachineState& state, const std::vector<int>& shape) {
    auto ops = input_operands(instr, state);
    size_t n = std::max(ops[0].data.size(), ops[1].data.size());
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) {
        float a = ops[0].data[i % ops[0].data.size()];
        float b = ops[1].data[i % ops[1].data.size()];
        out[i] = static_cast<float>(static_cast<float>(a) + static_cast<float>(b));
    }
    auto sh = shape.empty() ? (ops[0].shape.empty() ? ops[1].shape : ops[0].shape) : shape;
    return Value::from_array(std::move(out), sh, false);
}

static Value elem_muli(const Instr& instr, MachineState& state, const std::vector<int>& shape) {
    auto ops = input_operands(instr, state);
    size_t n = std::max(ops[0].data.size(), ops[1].data.size());
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) {
        int64_t a = static_cast<int64_t>(ops[0].data[i % ops[0].data.size()]);
        int64_t b = static_cast<int64_t>(ops[1].data[i % ops[1].data.size()]);
        out[i] = static_cast<float>(a * b);
    }
    auto sh = shape.empty() ? (ops[0].shape.empty() ? ops[1].shape : ops[0].shape) : shape;
    return Value::from_array(std::move(out), sh, true);
}

static Value elem_divsi(const Instr& instr, MachineState& state, const std::vector<int>& shape) {
    auto ops = input_operands(instr, state);
    size_t n = std::max(ops[0].data.size(), ops[1].data.size());
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) {
        int64_t a = static_cast<int64_t>(ops[0].data[i % ops[0].data.size()]);
        int64_t b = static_cast<int64_t>(ops[1].data[i % ops[1].data.size()]);
        out[i] = static_cast<float>(b != 0 ? a / b : 0);
    }
    auto sh = shape.empty() ? (ops[0].shape.empty() ? ops[1].shape : ops[0].shape) : shape;
    return Value::from_array(std::move(out), sh, true);
}

static Value elem_remsi(const Instr& instr, MachineState& state, const std::vector<int>& shape) {
    auto ops = input_operands(instr, state);
    size_t n = std::max(ops[0].data.size(), ops[1].data.size());
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) {
        int64_t a = static_cast<int64_t>(ops[0].data[i % ops[0].data.size()]);
        int64_t b = static_cast<int64_t>(ops[1].data[i % ops[1].data.size()]);
        out[i] = static_cast<float>(b != 0 ? a % b : 0);
    }
    auto sh = shape.empty() ? (ops[0].shape.empty() ? ops[1].shape : ops[0].shape) : shape;
    return Value::from_array(std::move(out), sh, true);
}

static Value elem_maxnumf(const Instr& instr, MachineState& state, const std::vector<int>& shape) {
    auto ops = input_operands(instr, state);
    size_t n = std::max(ops[0].data.size(), ops[1].data.size());
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) {
        float a = ops[0].data[i % ops[0].data.size()];
        float b = ops[1].data[i % ops[1].data.size()];
        out[i] = std::max(a, b);
    }
    auto sh = shape.empty() ? (ops[0].shape.empty() ? ops[1].shape : ops[0].shape) : shape;
    return Value::from_array(std::move(out), sh, false);
}

static Value elem_minnumf(const Instr& instr, MachineState& state, const std::vector<int>& shape) {
    auto ops = input_operands(instr, state);
    size_t n = std::max(ops[0].data.size(), ops[1].data.size());
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) {
        float a = ops[0].data[i % ops[0].data.size()];
        float b = ops[1].data[i % ops[1].data.size()];
        out[i] = std::min(a, b);
    }
    auto sh = shape.empty() ? (ops[0].shape.empty() ? ops[1].shape : ops[0].shape) : shape;
    return Value::from_array(std::move(out), sh, false);
}

static Value elem_mulf(const Instr& instr, MachineState& state, const std::vector<int>& shape) {
    auto ops = input_operands(instr, state);
    size_t n = std::max(ops[0].data.size(), ops[1].data.size());
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) {
        float a = ops[0].data[i % ops[0].data.size()];
        float b = ops[1].data[i % ops[1].data.size()];
        out[i] = a * b;
    }
    auto sh = shape.empty() ? (ops[0].shape.empty() ? ops[1].shape : ops[0].shape) : shape;
    return Value::from_array(std::move(out), sh, false);
}

static Value elem_subf(const Instr& instr, MachineState& state, const std::vector<int>& shape) {
    auto ops = input_operands(instr, state);
    size_t n = std::max(ops[0].data.size(), ops[1].data.size());
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) {
        float a = ops[0].data[i % ops[0].data.size()];
        float b = ops[1].data[i % ops[1].data.size()];
        out[i] = a - b;
    }
    auto sh = shape.empty() ? (ops[0].shape.empty() ? ops[1].shape : ops[0].shape) : shape;
    return Value::from_array(std::move(out), sh, false);
}

static Value elem_divf(const Instr& instr, MachineState& state, const std::vector<int>& shape) {
    auto ops = input_operands(instr, state);
    size_t n = std::max(ops[0].data.size(), ops[1].data.size());
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) {
        float a = ops[0].data[i % ops[0].data.size()];
        float b = ops[1].data[i % ops[1].data.size()];
        out[i] = a / b;
    }
    auto sh = shape.empty() ? (ops[0].shape.empty() ? ops[1].shape : ops[0].shape) : shape;
    return Value::from_array(std::move(out), sh, false);
}

// ---- Casts --------------------------------------------------------------- //
// The Python twin is exec.py::_extf/_truncf/_sitofp/_fptosi/_extsi/_trunci, and
// tests/test_schema_op_matrix.py checks both against independent NumPy expressions.
//
// Values are held as float32 here, so a cast whose result needs more than 24 bits of
// integer precision cannot be represented. Such a value is refused rather than silently
// rounded, because a silently rounded index is how an address goes wrong.

// f32 -> f16 -> f32, round-to-nearest-even, matching numpy's float16 cast.
// Rounds to the f16 grid by quantising: every f16 value is an integer multiple of a
// power-of-two quantum, 2^-24 in the subnormal range and 2^(exponent-10) above it.
// std::nearbyint rounds half-to-even under the default rounding mode, which is what
// IEEE 754 and numpy both use.
static float round_to_f16(float value) {
    if (!std::isfinite(value)) {
        return value;                               // Inf and NaN survive unchanged.
    }
    float magnitude = std::abs(value);
    if (magnitude > 65519.0f) {                     // Above the f16 round-to-max midpoint.
        return std::signbit(value) ? -std::numeric_limits<float>::infinity()
                                   :  std::numeric_limits<float>::infinity();
    }
    float quantum;
    if (magnitude < 6.103515625e-05f) {             // 2^-14: the subnormal range.
        quantum = 5.9604644775390625e-08f;          // 2^-24
    } else {
        int exponent;
        std::frexp(magnitude, &exponent);           // magnitude in [0.5, 1) * 2^exponent
        quantum = std::ldexp(1.0f, exponent - 11);  // 10 stored mantissa bits + implicit 1
    }
    float scaled = std::nearbyint(value / quantum);
    float result = scaled * quantum;
    if (std::abs(result) > 65504.0f) {              // Rounded up past the largest finite f16.
        return std::signbit(value) ? -std::numeric_limits<float>::infinity()
                                   :  std::numeric_limits<float>::infinity();
    }
    return result;
}

static void require_exact_integer(float value, const char* op) {
    if (!std::isfinite(value) || std::abs(value) > 16777216.0f) {
        throw UnsupportedInstruction(
            std::string(op) + ": operand does not fit the 24-bit exact integer range of the "
            "emulator's float32 storage; refusing rather than rounding an index");
    }
}

static Value cast_unary(const Instr& instr, MachineState& state, const std::vector<int>& shape,
                        const char* op, bool result_is_int) {
    auto ops = input_operands(instr, state);
    size_t n = ops[0].data.size();
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) {
        float v = ops[0].data[i];
        if (std::strcmp(op, "arith.truncf") == 0) {
            out[i] = round_to_f16(v);
        } else if (std::strcmp(op, "arith.fptosi") == 0) {
            require_exact_integer(std::trunc(v), op);
            out[i] = std::trunc(v);                 // toward zero, as MLIR specifies
        } else {
            require_exact_integer(v, op);
            out[i] = v;                             // widening casts are value-preserving
        }
    }
    auto sh = shape.empty() ? ops[0].shape : shape;
    return Value::from_array(std::move(out), sh, result_is_int);
}

static Value elem_extf(const Instr& i, MachineState& s, const std::vector<int>& sh) {
    return cast_unary(i, s, sh, "arith.extf", false);
}
static Value elem_truncf(const Instr& i, MachineState& s, const std::vector<int>& sh) {
    return cast_unary(i, s, sh, "arith.truncf", false);
}
static Value elem_sitofp(const Instr& i, MachineState& s, const std::vector<int>& sh) {
    return cast_unary(i, s, sh, "arith.sitofp", false);
}
static Value elem_fptosi(const Instr& i, MachineState& s, const std::vector<int>& sh) {
    return cast_unary(i, s, sh, "arith.fptosi", true);
}
static Value elem_extsi(const Instr& i, MachineState& s, const std::vector<int>& sh) {
    return cast_unary(i, s, sh, "arith.extsi", true);
}
static Value elem_trunci(const Instr& i, MachineState& s, const std::vector<int>& sh) {
    return cast_unary(i, s, sh, "arith.trunci", true);
}

static Value elem_negf(const Instr& instr, MachineState& state, const std::vector<int>& shape) {
    auto ops = input_operands(instr, state);
    size_t n = ops[0].data.size();
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) {
        out[i] = -ops[0].data[i];
    }
    auto sh = shape.empty() ? ops[0].shape : shape;
    return Value::from_array(std::move(out), sh, false);
}

static Value elem_absf(const Instr& instr, MachineState& state, const std::vector<int>& shape) {
    auto ops = input_operands(instr, state);
    size_t n = ops[0].data.size();
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) {
        out[i] = std::abs(ops[0].data[i]);
    }
    auto sh = shape.empty() ? ops[0].shape : shape;
    return Value::from_array(std::move(out), sh, false);
}

// arith.cmpi / arith.cmpf. The predicate is an integer `predicate` operand carrying the MLIR
// enum value (isa/predicates.py is the Python twin); a compare without one is refused, never
// defaulted. Unsigned predicates treat operands as 32-bit two's complement.
static int64_t compare_predicate(const Instr& instr, const MachineState& state, const std::string& op) {
    const Operand* operand = instr.operand("predicate");
    if (!operand) {
        throw UnsupportedInstruction(
            op + " carries no predicate operand; refusing rather than defaulting to slt");
    }
    return state.resolve(*operand).scalar_i();
}

static Value elem_cmpi(const Instr& instr, MachineState& state, const std::vector<int>& shape) {
    int64_t code = compare_predicate(instr, state, "arith.cmpi");
    auto ops = input_operands(instr, state);
    size_t n = std::max(ops[0].data.size(), ops[1].data.size());
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) {
        int64_t a = static_cast<int64_t>(ops[0].data[i % ops[0].data.size()]);
        int64_t b = static_cast<int64_t>(ops[1].data[i % ops[1].data.size()]);
        bool result = false;
        if (code >= 6 && code <= 9) {
            const int64_t lo = -(int64_t(1) << 31), hi = (int64_t(1) << 32) - 1;
            if (a < lo || a > hi || b < lo || b > hi) {
                throw UnsupportedInstruction("arith.cmpi: unsigned compare operand does not fit 32 bits");
            }
            uint32_t ua = static_cast<uint32_t>(a & 0xFFFFFFFF);
            uint32_t ub = static_cast<uint32_t>(b & 0xFFFFFFFF);
            switch (code) {
                case 6: result = ua < ub; break;
                case 7: result = ua <= ub; break;
                case 8: result = ua > ub; break;
                default: result = ua >= ub; break;
            }
        } else {
            switch (code) {
                case 0: result = a == b; break;
                case 1: result = a != b; break;
                case 2: result = a < b; break;
                case 3: result = a <= b; break;
                case 4: result = a > b; break;
                case 5: result = a >= b; break;
                default:
                    throw UnsupportedInstruction(
                        "arith.cmpi predicate code " + std::to_string(code) + " is not defined");
            }
        }
        out[i] = result ? 1.0f : 0.0f;
    }
    auto sh = shape.empty() ? (ops[0].shape.empty() ? ops[1].shape : ops[0].shape) : shape;
    return Value::from_array(std::move(out), sh, true);
}

static Value elem_cmpf(const Instr& instr, MachineState& state, const std::vector<int>& shape) {
    int64_t code = compare_predicate(instr, state, "arith.cmpf");
    auto ops = input_operands(instr, state);
    size_t n = std::max(ops[0].data.size(), ops[1].data.size());
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) {
        float a = ops[0].data[i % ops[0].data.size()];
        float b = ops[1].data[i % ops[1].data.size()];
        bool un = std::isnan(a) || std::isnan(b);
        bool result = false;
        switch (code) {
            case 0: result = false; break;
            case 1: result = !un && a == b; break;
            case 2: result = !un && a > b; break;
            case 3: result = !un && a >= b; break;
            case 4: result = !un && a < b; break;
            case 5: result = !un && a <= b; break;
            case 6: result = !un && a != b; break;
            case 7: result = !un; break;
            case 8: result = un || a == b; break;
            case 9: result = un || a > b; break;
            case 10: result = un || a >= b; break;
            case 11: result = un || a < b; break;
            case 12: result = un || a <= b; break;
            case 13: result = un || a != b; break;
            case 14: result = un; break;
            case 15: result = true; break;
            default:
                throw UnsupportedInstruction(
                    "arith.cmpf predicate code " + std::to_string(code) + " is not defined");
        }
        out[i] = result ? 1.0f : 0.0f;
    }
    auto sh = shape.empty() ? (ops[0].shape.empty() ? ops[1].shape : ops[0].shape) : shape;
    return Value::from_array(std::move(out), sh, true);
}

static Value elem_addptr(const Instr& instr, MachineState& state, const std::vector<int>& shape) {
    auto ops = input_operands(instr, state);
    size_t n = std::max(ops[0].data.size(), ops[1].data.size());
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) {
        int64_t a = static_cast<int64_t>(ops[0].data[i % ops[0].data.size()]);
        int64_t b = static_cast<int64_t>(ops[1].data[i % ops[1].data.size()]);
        out[i] = static_cast<float>(a + b);
    }
    auto sh = shape.empty() ? (ops[0].shape.empty() ? ops[1].shape : ops[0].shape) : shape;
    return Value::from_array(std::move(out), sh, true);
}

static Value elem_subi(const Instr& instr, MachineState& state, const std::vector<int>& shape) {
    auto ops = input_operands(instr, state);
    size_t n = std::max(ops[0].data.size(), ops[1].data.size());
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) {
        int64_t a = static_cast<int64_t>(ops[0].data[i % ops[0].data.size()]);
        int64_t b = static_cast<int64_t>(ops[1].data[i % ops[1].data.size()]);
        out[i] = static_cast<float>(a - b);
    }
    auto sh = shape.empty() ? (ops[0].shape.empty() ? ops[1].shape : ops[0].shape) : shape;
    return Value::from_array(std::move(out), sh, true);
}

static Value elem_divui(const Instr& instr, MachineState& state, const std::vector<int>& shape) {
    auto ops = input_operands(instr, state);
    size_t n = std::max(ops[0].data.size(), ops[1].data.size());
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) {
        uint64_t a = static_cast<uint64_t>(static_cast<int64_t>(ops[0].data[i % ops[0].data.size()]));
        uint64_t b = static_cast<uint64_t>(static_cast<int64_t>(ops[1].data[i % ops[1].data.size()]));
        out[i] = static_cast<float>(static_cast<int64_t>(b != 0 ? a / b : 0));
    }
    auto sh = shape.empty() ? (ops[0].shape.empty() ? ops[1].shape : ops[0].shape) : shape;
    return Value::from_array(std::move(out), sh, true);
}

static Value elem_remui(const Instr& instr, MachineState& state, const std::vector<int>& shape) {
    auto ops = input_operands(instr, state);
    size_t n = std::max(ops[0].data.size(), ops[1].data.size());
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) {
        uint64_t a = static_cast<uint64_t>(static_cast<int64_t>(ops[0].data[i % ops[0].data.size()]));
        uint64_t b = static_cast<uint64_t>(static_cast<int64_t>(ops[1].data[i % ops[1].data.size()]));
        out[i] = static_cast<float>(static_cast<int64_t>(b != 0 ? a % b : 0));
    }
    auto sh = shape.empty() ? (ops[0].shape.empty() ? ops[1].shape : ops[0].shape) : shape;
    return Value::from_array(std::move(out), sh, true);
}

static Value elem_maxsi(const Instr& instr, MachineState& state, const std::vector<int>& shape) {
    auto ops = input_operands(instr, state);
    size_t n = std::max(ops[0].data.size(), ops[1].data.size());
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) {
        int64_t a = static_cast<int64_t>(ops[0].data[i % ops[0].data.size()]);
        int64_t b = static_cast<int64_t>(ops[1].data[i % ops[1].data.size()]);
        out[i] = static_cast<float>(std::max(a, b));
    }
    auto sh = shape.empty() ? (ops[0].shape.empty() ? ops[1].shape : ops[0].shape) : shape;
    return Value::from_array(std::move(out), sh, true);
}

static Value elem_minsi(const Instr& instr, MachineState& state, const std::vector<int>& shape) {
    auto ops = input_operands(instr, state);
    size_t n = std::max(ops[0].data.size(), ops[1].data.size());
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) {
        int64_t a = static_cast<int64_t>(ops[0].data[i % ops[0].data.size()]);
        int64_t b = static_cast<int64_t>(ops[1].data[i % ops[1].data.size()]);
        out[i] = static_cast<float>(std::min(a, b));
    }
    auto sh = shape.empty() ? (ops[0].shape.empty() ? ops[1].shape : ops[0].shape) : shape;
    return Value::from_array(std::move(out), sh, true);
}

static Value elem_maxui(const Instr& instr, MachineState& state, const std::vector<int>& shape) {
    auto ops = input_operands(instr, state);
    size_t n = std::max(ops[0].data.size(), ops[1].data.size());
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) {
        uint64_t a = static_cast<uint64_t>(static_cast<int64_t>(ops[0].data[i % ops[0].data.size()]));
        uint64_t b = static_cast<uint64_t>(static_cast<int64_t>(ops[1].data[i % ops[1].data.size()]));
        out[i] = static_cast<float>(static_cast<int64_t>(std::max(a, b)));
    }
    auto sh = shape.empty() ? (ops[0].shape.empty() ? ops[1].shape : ops[0].shape) : shape;
    return Value::from_array(std::move(out), sh, true);
}

static Value elem_minui(const Instr& instr, MachineState& state, const std::vector<int>& shape) {
    auto ops = input_operands(instr, state);
    size_t n = std::max(ops[0].data.size(), ops[1].data.size());
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) {
        uint64_t a = static_cast<uint64_t>(static_cast<int64_t>(ops[0].data[i % ops[0].data.size()]));
        uint64_t b = static_cast<uint64_t>(static_cast<int64_t>(ops[1].data[i % ops[1].data.size()]));
        out[i] = static_cast<float>(static_cast<int64_t>(std::min(a, b)));
    }
    auto sh = shape.empty() ? (ops[0].shape.empty() ? ops[1].shape : ops[0].shape) : shape;
    return Value::from_array(std::move(out), sh, true);
}

static Value elem_clamp(const Instr& instr, MachineState& state, const std::vector<int>& shape) {
    auto ops = input_operands(instr, state);
    if (ops.empty()) {
        throw UnsupportedInstruction("clamp requires operands");
    }
    float lo = 0.0f;
    float hi = 0.0f;
    bool has_bounds = false;
    if (ops.size() >= 3) {
        lo = ops[1].data[0];
        hi = ops[2].data[0];
        has_bounds = true;
    } else {
        auto it_lo = instr.operands.find("lo");
        auto it_hi = instr.operands.find("hi");
        if (it_lo != instr.operands.end() && it_hi != instr.operands.end()) {
            lo = state.resolve(it_lo->second).scalar_f();
            hi = state.resolve(it_hi->second).scalar_f();
            has_bounds = true;
        }
    }
    if (!has_bounds) {
        throw UnsupportedInstruction("clamp requires lo and hi operands");
    }
    size_t n = ops[0].data.size();
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) {
        float x = ops[0].data[i];
        out[i] = std::min(std::max(x, lo), hi);
    }
    auto sh = shape.empty() ? ops[0].shape : shape;
    return Value::from_array(std::move(out), sh, false);
}

// Dispatch table.
static const std::map<std::string, ElemHandler> ELEMENTWISE_OPS = {
    {"arith.constant", elem_const},
    {"constant", elem_const},
    {"tt.get_program_id", elem_program_id},
    {"get_program_id", elem_program_id},
    {"tt.make_range", elem_make_range},
    {"make_range", elem_make_range},
    {"tt.splat", elem_splat},
    {"splat", elem_splat},
    {"tt.broadcast", elem_broadcast},
    {"broadcast", elem_broadcast},
    {"tt.expand_dims", elem_expand_dims},
    {"expand_dims", elem_expand_dims},
    // Add
    {"arith.addi", elem_addi},
    {"addi", elem_addi},
    {"arith.addf", elem_addf},
    {"addf", elem_addf},
    // Sub
    {"arith.subi", elem_subi},
    {"subi", elem_subi},
    {"arith.subf", elem_subf},
    {"subf", elem_subf},
    // Mul
    {"arith.muli", elem_muli},
    {"muli", elem_muli},
    {"arith.mulf", elem_mulf},
    {"mulf", elem_mulf},
    // Div
    {"arith.divsi", elem_divsi},
    {"divsi", elem_divsi},
    {"arith.divui", elem_divui},
    {"divui", elem_divui},
    {"arith.divf", elem_divf},
    {"divf", elem_divf},
    // Rem
    {"arith.remsi", elem_remsi},
    {"remsi", elem_remsi},
    {"arith.remui", elem_remui},
    {"remui", elem_remui},
    // Max / Min
    {"arith.maxnumf", elem_maxnumf},
    {"maxnumf", elem_maxnumf},
    {"arith.minnumf", elem_minnumf},
    {"minnumf", elem_minnumf},
    {"arith.maxsi", elem_maxsi},
    {"maxsi", elem_maxsi},
    {"arith.minsi", elem_minsi},
    {"minsi", elem_minsi},
    {"arith.maxui", elem_maxui},
    {"maxui", elem_maxui},
    {"arith.minui", elem_minui},
    {"minui", elem_minui},
    // Neg / Abs
    {"arith.negf", elem_negf},
    {"negf", elem_negf},
    {"math.absf", elem_absf},
    {"arith.absf", elem_absf},
    {"absf", elem_absf},
    // Clamp
    {"tt.clamp", elem_clamp},
    {"clamp", elem_clamp},
    // Other
    {"arith.cmpi", elem_cmpi},
    {"cmpi", elem_cmpi},
    {"arith.cmpf", elem_cmpf},
    {"cmpf", elem_cmpf},
    // Casts
    {"arith.extf", elem_extf},
    {"extf", elem_extf},
    {"arith.truncf", elem_truncf},
    {"truncf", elem_truncf},
    {"arith.sitofp", elem_sitofp},
    {"sitofp", elem_sitofp},
    {"arith.fptosi", elem_fptosi},
    {"fptosi", elem_fptosi},
    {"arith.extsi", elem_extsi},
    {"extsi", elem_extsi},
    {"arith.trunci", elem_trunci},
    {"trunci", elem_trunci},
    {"tt.addptr", elem_addptr},
    {"addptr", elem_addptr},
};

static void apply_elementwise(const Instr& instr, MachineState& state) {
    if (instr.name == "LI" || instr.name == "MOVI") {
        if (instr.defs.empty()) {
            throw UnsupportedInstruction(instr.name + " defines no value to bind");
        }
        auto shape = declared_shape(instr);
        state.bind(instr.defs[0], elem_const(instr, state, shape));
        return;
    }
    std::string op;
    if (instr.source) {
        op = instr.source->op_name;
    }
    if (op.empty()) {
        throw UnsupportedInstruction(
            instr.name + " carries no source operation; the elementwise "
            "unit cannot know which arithmetic to perform");
    }

    auto shape = declared_shape(instr);

    auto it = ELEMENTWISE_OPS.find(op);
    if (it == ELEMENTWISE_OPS.end()) {
        throw UnsupportedInstruction(
            "elementwise operation '" + op + "' is not implemented by this machine; "
            "refusing rather than substituting");
    }

    if (instr.defs.empty()) {
        throw UnsupportedInstruction(op + " defines no value to bind");
    }
    Value result = it->second(instr, state, shape);
    state.bind(instr.defs[0], std::move(result));
}

// ---- Loop execution ----------------------------------------------------- //

void run_loop(const Loop& loop, MachineState& state, const PrecisionPolicy& policy) {
    if (!loop.yields.empty() &&
        loop.yields.size() != loop.iter_args.size()) {
        throw StorageError(
            "loop yields " + std::to_string(loop.yields.size()) +
            " value(s) for " + std::to_string(loop.iter_args.size()) +
            " iter_args; cannot re-thread");
    }
    if (loop.inits.size() != loop.iter_args.size()) {
        throw StorageError(
            "loop has " + std::to_string(loop.inits.size()) +
            " initialiser(s) for " + std::to_string(loop.iter_args.size()) +
            " iter_args; cannot enter the loop");
    }

    // Bind iter_args from initialisers.
    for (size_t i = 0; i < loop.iter_args.size(); ++i) {
        Value init_val = state.resolve(SsaRef{loop.inits[i]});
        state.bind(loop.iter_args[i], std::move(init_val));
    }

    // Resolve loop bounds.
    int lower = 0, upper = 0, step = 1;
    if (loop.lower) {
        lower = static_cast<int>(state.resolve(SsaRef{*loop.lower}).scalar_i());
    }
    if (loop.upper) {
        upper = static_cast<int>(state.resolve(SsaRef{*loop.upper}).scalar_i());
    }
    if (loop.step) {
        step = static_cast<int>(state.resolve(SsaRef{*loop.step}).scalar_i());
    }
    if (step == 0) {
        throw StorageError("loop has step 0; it would never terminate");
    }

    int index = lower;
    int guard = (upper - lower) / step + 2;
    int iteration = 0;

    while (index < upper) {
        state.loop_iteration = iteration;
        --guard;
        if (guard < 0) {
            throw StorageError("loop did not terminate; refusing to spin");
        }

        if (loop.induction_var) {
            state.bind(*loop.induction_var, Value::from_int(index));
        }

        // Execute body.
        for (const auto& item : loop.body) {
            if (auto* instr = std::get_if<Instr>(&item)) {
                apply(*instr, state, policy);
            }
            // UnsupportedMarker in body: handled by the markers() check
            // before we enter execution.
        }

        // Re-thread iter_args from yields.
        if (!loop.yields.empty()) {
            std::vector<Value> advanced;
            for (const auto& name : loop.yields) {
                advanced.push_back(state.resolve(SsaRef{name}));
            }
            for (size_t i = 0; i < loop.iter_args.size(); ++i) {
                state.bind(loop.iter_args[i], std::move(advanced[i]));
            }
        }

        index += step;
        ++iteration;
    }

    // Bind results from final iter_args.
    for (size_t i = 0; i < loop.results.size() && i < loop.iter_args.size(); ++i) {
        auto it = state.values.find(loop.iter_args[i]);
        if (it != state.values.end()) {
            state.bind(loop.results[i], it->second);
        }
    }
}

// ---- Top-level emulate -------------------------------------------------- //

std::map<std::string, Value> emulate(
    const Program& program,
    const std::map<std::string, Value>& inputs,
    const PrecisionPolicy& policy,
    int gx, int gy, int gz
) {
    // Postcondition 2: UNSUPPORTED halts locally.
    auto markers = program.markers();
    if (!markers.empty()) {
        throw ProgramNotExecutable(
            markers[0]->op_name,
            markers[0]->reason,
            markers[0]->loc_name.value_or(""));
    }

    MachineState state = MachineState::from_program(program, inputs, policy, gx, gy, gz);

    for (const auto& item : program.items) {
        if (auto* loop = std::get_if<Loop>(&item)) {
            run_loop(*loop, state, policy);
        } else if (auto* instr = std::get_if<Instr>(&item)) {
            apply(*instr, state, policy);
        }
        // UnsupportedMarker already caught above.
    }

    return state.outputs();
}

}  // namespace tritonflow::emu
