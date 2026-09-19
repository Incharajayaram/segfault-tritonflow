// ir_types.h — C++ representations of the program IR from emit/ir.py.
//
// These are the data structures the emulator consumes. They mirror
// data-model.md §5 and emit/ir.py's SourceRef, SsaRef, Imm, MemRef,
// Instr, Loop, UnsupportedMarker, and Program.
//
// The program model is name-based: operands hold SSA name strings,
// not object references. This is what makes serialisation possible.

#pragma once

#include <cmath>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace tritonflow::emu {

// ---- Source identity ---------------------------------------------------- //

/// Where an instruction came from: the operation, its position, its loc.
struct SourceRef {
    std::string op_name;
    int line = 0;
    int col = 0;
    std::optional<std::string> loc_name;
};

// ---- Operands ----------------------------------------------------------- //

/// A value produced inside the program (or declared in Program.inputs).
struct SsaRef {
    std::string name;
};

/// A literal. int and float are distinguished on purpose (EC-082).
struct Imm {
    std::variant<int64_t, double> value;

    bool is_int() const {
        return std::holds_alternative<int64_t>(value);
    }

    double as_double() const {
        if (auto* i = std::get_if<int64_t>(&value)) return static_cast<double>(*i);
        return std::get<double>(value);
    }

    int64_t as_int() const {
        if (auto* i = std::get_if<int64_t>(&value)) return *i;
        return static_cast<int64_t>(std::get<double>(value));
    }
};

/// A memory reference: a space, a base, and the addressing decision.
struct MemRef {
    std::string space;
    std::string base;
    std::optional<std::string> access_key;
};

/// data-model.md §5: Operand = SsaRef | Imm | MemRef.
using Operand = std::variant<SsaRef, Imm, MemRef>;

// ---- Marker kinds ------------------------------------------------------- //

enum class MarkerKind {
    UNSUPPORTED,
    PARSE_UNSUPPORTED,
};

/// An operation we chose not to, or could not, lower (FR-005).
struct UnsupportedMarker {
    std::string op_name;
    std::string reason;
    std::optional<std::string> loc_name;
    MarkerKind kind = MarkerKind::UNSUPPORTED;
    std::optional<SourceRef> source;
};

// ---- Instructions ------------------------------------------------------- //

/// One instruction in the emitted program.
struct Instr {
    /// Schema instruction name (e.g. "DMA1D", "MAC8", "EPI").
    std::string name;

    /// role -> operand, e.g. {"acc": ..., "a": ..., "b": ...} for a MAC.
    std::map<std::string, Operand> operands;

    /// The roles in order (determines _operands() traversal for elementwise).
    std::vector<std::string> roles;

    /// Names this instruction produces.
    std::vector<std::string> defs;

    /// The source TTIR operation this instruction came from.
    std::optional<SourceRef> source;

    /// The descriptor key this instruction was constrained against.
    std::optional<std::string> constrained_on;

    /// Cost from the selector.
    double cost = 0.0;

    /// Get an operand by role, or nullptr if not present.
    const Operand* operand(const std::string& role) const {
        auto it = operands.find(role);
        if (it == operands.end()) return nullptr;
        return &it->second;
    }
};

// ---- Loops -------------------------------------------------------------- //

/// A recovered scf.for loop.
struct Loop {
    int id = 0;

    /// The induction variable name (e.g. "_k").
    std::optional<std::string> induction_var;

    /// Loop bounds — names of SSA values or immediates.
    std::optional<std::string> lower;
    std::optional<std::string> upper;
    std::optional<std::string> step;

    /// iter_args names — the loop-carried values.
    std::vector<std::string> iter_args;

    /// Initialiser names for each iter_arg.
    std::vector<std::string> inits;

    /// Yield names — which values feed the next iteration's iter_args.
    std::vector<std::string> yields;

    /// Result names — what a post-loop reader sees.
    std::vector<std::string> results;

    /// The body: instructions and markers in execution order.
    std::vector<std::variant<Instr, UnsupportedMarker>> body;
};

// ---- Program ------------------------------------------------------------ //

/// An item in the program's top-level execution order.
using ProgramItem = std::variant<Instr, Loop, UnsupportedMarker>;

/// The complete emitted program.
struct Program {
    /// Declared inputs — the names the caller must supply.
    std::vector<std::string> inputs;

    /// Top-level items in execution order.
    std::vector<ProgramItem> items;

    /// Schema version this program was emitted against.
    int schema_version = 1;

    /// Total cost (fsum of instruction costs).
    double total_cost = 0.0;

    /// Per-instruction legal (source space, destination space) moves, supplied by
    /// the caller from the ISA schema. The C++ emulator has no schema loader, so an
    /// instruction absent from this table cannot have a memory access verified and
    /// is refused rather than let through.
    std::map<std::string, std::vector<std::pair<std::string, std::string>>> transfers;

    /// Collect all UNSUPPORTED markers in the program.
    std::vector<const UnsupportedMarker*> markers() const {
        std::vector<const UnsupportedMarker*> result;
        for (const auto& item : items) {
            if (auto* m = std::get_if<UnsupportedMarker>(&item)) {
                result.push_back(m);
            }
            if (auto* loop = std::get_if<Loop>(&item)) {
                for (const auto& bi : loop->body) {
                    if (auto* m = std::get_if<UnsupportedMarker>(&bi)) {
                        result.push_back(m);
                    }
                }
            }
        }
        return result;
    }

    /// Iterate all instructions (top-level and inside loops).
    std::vector<const Instr*> instructions() const {
        std::vector<const Instr*> result;
        for (const auto& item : items) {
            if (auto* instr = std::get_if<Instr>(&item)) {
                result.push_back(instr);
            }
            if (auto* loop = std::get_if<Loop>(&item)) {
                for (const auto& bi : loop->body) {
                    if (auto* instr = std::get_if<Instr>(&bi)) {
                        result.push_back(instr);
                    }
                }
            }
        }
        return result;
    }
};

}  // namespace tritonflow::emu
