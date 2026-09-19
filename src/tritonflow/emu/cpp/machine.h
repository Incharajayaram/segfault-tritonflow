// machine.h — MachineState, Storage, and the emulate/apply interface.
//
// Mirrors exec.py. Contract: contracts/emulator.md.
//
// Memory is one flat fp32 array; a pointer is an element index into it.
// A value is a float scalar, a 1-D vector, or a 2-D matrix.
// The instruction's own source.op_name disambiguates the overloaded EPI.

#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "errors.h"
#include "ir_types.h"
#include "precision.h"

namespace tritonflow::emu {

// ---- Value type --------------------------------------------------------- //

/// A runtime value in the machine: scalar int, scalar float, or a shaped array.
struct Value {
    std::vector<float> data;
    std::vector<int> shape;  // empty = scalar
    bool is_int = false;     // whether the source was integer-typed

    // -- Scalar constructors ------------------------------------------------
    static Value from_int(int64_t v) {
        Value val;
        val.data = {static_cast<float>(v)};
        val.is_int = true;
        return val;
    }

    static Value from_float(double v) {
        Value val;
        val.data = {static_cast<float>(v)};
        val.is_int = false;
        return val;
    }

    static Value from_array(std::vector<float> data, std::vector<int> shape, bool is_int = false) {
        Value val;
        val.data = std::move(data);
        val.shape = std::move(shape);
        val.is_int = is_int;
        return val;
    }

    bool is_scalar() const { return shape.empty(); }

    float scalar_f() const { return data.empty() ? 0.0f : data[0]; }
    int64_t scalar_i() const { return static_cast<int64_t>(scalar_f()); }

    size_t size() const { return data.size(); }

    /// Total element count implied by shape (or 1 if scalar).
    size_t numel() const {
        if (shape.empty()) return 1;
        size_t n = 1;
        for (int d : shape) n *= static_cast<size_t>(d);
        return n;
    }
};

// ---- Storage ------------------------------------------------------------ //

/// One named buffer: where it lives and what shape it came in as.
struct Storage {
    std::string name;
    int base = 0;
    int length = 0;
    std::vector<int> shape;

    int end() const { return base + length; }
};

// ---- MachineState ------------------------------------------------------- //

/// The toy machine: flat memory, the named buffers, and the value store.
class MachineState {
public:
    const Program* program = nullptr;
    PrecisionPolicy policy;
    std::vector<float> memory;
    std::map<std::string, Storage> storages;
    std::unordered_map<std::string, Value> values;
    std::set<std::string> written;
    int grid[3] = {0, 0, 0};
    int loop_iteration = 0;

    // -- Construction ------------------------------------------------------ //

    /// Build a MachineState from a program and its inputs.
    /// `inputs` maps input names to flat float arrays with associated shapes.
    static MachineState from_program(
        const Program& program,
        const std::map<std::string, Value>& inputs,
        const PrecisionPolicy& policy,
        int gx = 0, int gy = 0, int gz = 0);

    // -- Values ------------------------------------------------------------ //

    /// Resolve an operand to a Value.
    Value resolve(const Operand& operand) const;

    /// Bind a name to a value.
    void bind(const std::string& name, Value value);

    // -- Memory ------------------------------------------------------------ //

    /// Gather: read from flat memory at the given indices.
    Value gather(const Value& indices, const Value* mask) const;

    /// Scatter: write to flat memory at the given indices.
    void scatter(const Value& indices, const Value& values, const Value* mask);

    /// Return every buffer the program wrote, with its original shape.
    std::map<std::string, Value> outputs() const;

private:
    void check_bounds(const std::vector<int64_t>& flat) const;
    void mark_written(const std::vector<int64_t>& flat);
};

// ---- Instruction execution ---------------------------------------------- //

/// Execute one instruction against state (contracts/emulator.md interface).
void apply(const Instr& instr, MachineState& state, const PrecisionPolicy& policy);

/// Execute one recovered scf.for, re-threading iter_args through yields.
void run_loop(const Loop& loop, MachineState& state, const PrecisionPolicy& policy);

/// Execute a program and return every buffer it wrote (postcondition 1).
std::map<std::string, Value> emulate(
    const Program& program,
    const std::map<std::string, Value>& inputs,
    const PrecisionPolicy& policy,
    int gx = 0, int gy = 0, int gz = 0);

}  // namespace tritonflow::emu
