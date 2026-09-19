// bindings.cpp — pybind11 module `_emu_cpp`.
//
// Exposes the C++ emulator to Python. Accepts Python dicts (from
// Program.serialize()) and NumPy arrays for inputs, returns NumPy
// arrays for outputs.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "errors.h"
#include "ir_types.h"
#include "machine.h"
#include "precision.h"

namespace py = pybind11;
using namespace tritonflow::emu;

// ---- Conversion helpers ------------------------------------------------- //

/// Convert a Python dict of numpy arrays into C++ Value inputs.
static std::map<std::string, Value> convert_inputs(const py::dict& inputs) {
    std::map<std::string, Value> result;
    for (auto& [key, val] : inputs) {
        std::string name = py::cast<std::string>(key);
        if (py::isinstance<py::array>(val)) {
            auto arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(val);
            auto buf = arr.request();
            std::vector<float> data(
                static_cast<float*>(buf.ptr),
                static_cast<float*>(buf.ptr) + buf.size);
            std::vector<int> shape;
            for (auto d : buf.shape) shape.push_back(static_cast<int>(d));
            result[name] = Value::from_array(std::move(data), std::move(shape));
        } else if (py::isinstance<py::int_>(val)) {
            result[name] = Value::from_int(py::cast<int64_t>(val));
        } else if (py::isinstance<py::float_>(val)) {
            result[name] = Value::from_float(py::cast<double>(val));
        } else {
            // Try to convert to float.
            result[name] = Value::from_float(py::cast<double>(val));
        }
    }
    return result;
}

/// Convert a Python Operand (from ir.py serialization) to C++ Operand.
static Operand convert_operand(const py::object& obj) {
    if (py::hasattr(obj, "space") && py::isinstance<py::str>(obj.attr("space"))) {
        MemRef m;
        m.space = py::cast<std::string>(obj.attr("space"));
        m.base = py::cast<std::string>(obj.attr("base"));
        if (py::hasattr(obj, "access_key") && !obj.attr("access_key").is_none()) {
            m.access_key = py::cast<std::string>(obj.attr("access_key"));
        }
        return m;
    }
    if (py::hasattr(obj, "name") && py::isinstance<py::str>(obj.attr("name"))) {
        return SsaRef{py::cast<std::string>(obj.attr("name"))};
    }
    if (py::hasattr(obj, "value")) {
        auto v = obj.attr("value");
        if (py::isinstance<py::int_>(v)) {
            return Imm{py::cast<int64_t>(v)};
        }
        return Imm{py::cast<double>(v)};
    }
    throw std::runtime_error("cannot convert operand");
}

/// Convert a Python SourceRef to C++.
static std::optional<SourceRef> convert_source(const py::object& obj) {
    if (obj.is_none()) return std::nullopt;
    SourceRef s;
    s.op_name = py::cast<std::string>(obj.attr("op_name"));
    s.line = py::cast<int>(obj.attr("line"));
    s.col = py::cast<int>(obj.attr("col"));
    if (py::hasattr(obj, "loc_name") && !obj.attr("loc_name").is_none()) {
        s.loc_name = py::cast<std::string>(obj.attr("loc_name"));
    }
    return s;
}

/// Convert a Python Instr to C++.
static Instr convert_instr(const py::object& obj) {
    Instr instr;
    instr.name = py::cast<std::string>(obj.attr("name"));

    // Operands.
    auto ops = py::cast<py::dict>(obj.attr("operands"));
    for (auto& [key, val] : ops) {
        instr.operands[py::cast<std::string>(key)] = convert_operand(py::cast<py::object>(val));
    }

    // Roles.
    auto roles = py::cast<py::list>(obj.attr("roles"));
    for (auto& r : roles) {
        instr.roles.push_back(py::cast<std::string>(r));
    }

    // Defs.
    auto defs = py::cast<py::list>(obj.attr("defs"));
    for (auto& d : defs) {
        instr.defs.push_back(py::cast<std::string>(d));
    }

    // Source.
    if (py::hasattr(obj, "source")) {
        instr.source = convert_source(obj.attr("source"));
    }

    // Constrained_on.
    if (py::hasattr(obj, "constrained_on") && !obj.attr("constrained_on").is_none()) {
        instr.constrained_on = py::cast<std::string>(obj.attr("constrained_on"));
    }

    // Cost.
    if (py::hasattr(obj, "cost")) {
        instr.cost = py::cast<double>(obj.attr("cost"));
    }

    return instr;
}

/// Convert a Python UnsupportedMarker to C++.
static UnsupportedMarker convert_marker(const py::object& obj) {
    UnsupportedMarker m;
    m.op_name = py::cast<std::string>(obj.attr("op_name"));
    m.reason = py::cast<std::string>(obj.attr("reason"));
    if (py::hasattr(obj, "loc_name") && !obj.attr("loc_name").is_none()) {
        m.loc_name = py::cast<std::string>(obj.attr("loc_name"));
    }
    auto kind_str = py::cast<std::string>(obj.attr("kind"));
    m.kind = (kind_str == "PARSE_UNSUPPORTED") ? MarkerKind::PARSE_UNSUPPORTED
                                                : MarkerKind::UNSUPPORTED;
    if (py::hasattr(obj, "source")) {
        m.source = convert_source(obj.attr("source"));
    }
    return m;
}

/// Convert a Python Loop to C++.
static Loop convert_loop(const py::object& obj) {
    Loop loop;
    loop.id = py::cast<int>(obj.attr("id"));

    if (py::hasattr(obj, "induction_var") && !obj.attr("induction_var").is_none()) {
        loop.induction_var = py::cast<std::string>(obj.attr("induction_var"));
    }
    if (py::hasattr(obj, "lower") && !obj.attr("lower").is_none()) {
        loop.lower = py::cast<std::string>(py::str(obj.attr("lower")));
    }
    if (py::hasattr(obj, "upper") && !obj.attr("upper").is_none()) {
        loop.upper = py::cast<std::string>(py::str(obj.attr("upper")));
    }
    if (py::hasattr(obj, "step") && !obj.attr("step").is_none()) {
        loop.step = py::cast<std::string>(py::str(obj.attr("step")));
    }

    for (auto& s : py::cast<py::list>(obj.attr("iter_args"))) {
        loop.iter_args.push_back(py::cast<std::string>(s));
    }
    for (auto& s : py::cast<py::list>(obj.attr("inits"))) {
        loop.inits.push_back(py::cast<std::string>(s));
    }
    for (auto& s : py::cast<py::list>(obj.attr("yields"))) {
        loop.yields.push_back(py::cast<std::string>(s));
    }
    for (auto& s : py::cast<py::list>(obj.attr("results"))) {
        loop.results.push_back(py::cast<std::string>(s));
    }

    for (auto& item : py::cast<py::list>(obj.attr("body"))) {
        py::object it = py::cast<py::object>(item);
        if (py::hasattr(it, "roles")) {
            loop.body.push_back(convert_instr(it));
        } else {
            loop.body.push_back(convert_marker(it));
        }
    }

    return loop;
}

/// Convert a Python Program to C++.
static Program convert_program(const py::object& obj) {
    Program prog;

    for (auto& s : py::cast<py::list>(obj.attr("inputs"))) {
        prog.inputs.push_back(py::cast<std::string>(s));
    }

    if (py::hasattr(obj, "schema_version")) {
        prog.schema_version = py::cast<int>(obj.attr("schema_version"));
    }

    // execution_order() returns the items in execution order.
    auto exec_order = obj.attr("execution_order")();
    for (auto& item : py::cast<py::list>(exec_order)) {
        py::object it = py::cast<py::object>(item);
        auto type_name = py::cast<std::string>(py::str(py::type::of(it).attr("__name__")));
        if (type_name == "Loop") {
            prog.items.push_back(convert_loop(it));
        } else if (type_name == "Instr") {
            prog.items.push_back(convert_instr(it));
        } else if (type_name == "UnsupportedMarker") {
            prog.items.push_back(convert_marker(it));
        }
    }

    // Also check for markers (e.g. from program.unsupported).
    for (auto item : obj.attr("markers")()) {
        py::object it = py::reinterpret_borrow<py::object>(item);
        prog.items.push_back(convert_marker(it));
    }

    return prog;
}

/// Convert C++ outputs to Python dict of numpy arrays.
static py::dict convert_outputs(const std::map<std::string, Value>& outputs) {
    py::dict result;
    for (const auto& [name, val] : outputs) {
        std::vector<py::ssize_t> shape;
        for (int d : val.shape) shape.push_back(d);
        if (shape.empty()) shape.push_back(static_cast<py::ssize_t>(val.data.size()));

        py::array_t<float> arr(shape);
        auto buf = arr.request();
        std::memcpy(buf.ptr, val.data.data(), val.data.size() * sizeof(float));
        result[py::cast(name)] = arr;
    }
    return result;
}

// ---- Module definition -------------------------------------------------- //

PYBIND11_MODULE(_emu_cpp, m) {
    m.doc() = "C++ emulator backend for tritonflow (contracts/emulator.md)";

    // ---- Exceptions ----
    static py::exception<ProgramNotExecutable> exc_pne(m, "ProgramNotExecutable");
    static py::exception<UnsupportedInstruction> exc_ui(m, "UnsupportedInstruction");
    static py::exception<StorageError> exc_se(m, "StorageError");
    static py::exception<MissingInput> exc_mi(m, "MissingInput");
    static py::exception<ShapeMismatch> exc_sm(m, "ShapeMismatch");

    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) std::rethrow_exception(p);
        } catch (const ProgramNotExecutable& e) {
            exc_pne(e.what());
        } catch (const UnsupportedInstruction& e) {
            exc_ui(e.what());
        } catch (const StorageError& e) {
            exc_se(e.what());
        } catch (const MissingInput& e) {
            exc_mi(e.what());
        } catch (const ShapeMismatch& e) {
            exc_sm(e.what());
        }
    });

    // ---- Tolerance ----
    py::class_<Tolerance>(m, "Tolerance")
        .def_readonly("value", &Tolerance::value)
        .def_readonly("derivation", &Tolerance::derivation)
        .def_readonly("absolute", &Tolerance::absolute);

    // ---- Comparison ----
    py::class_<Comparison>(m, "Comparison")
        .def_readonly("ok", &Comparison::ok)
        .def_readonly("max_error", &Comparison::max_error)
        .def_readonly("bound_value", &Comparison::bound_value)
        .def_readonly("derivation", &Comparison::derivation);

    // ---- PrecisionPolicy ----
    py::class_<PrecisionPolicy>(m, "PrecisionPolicy")
        .def(py::init<>())
        .def(py::init([](const std::string& prec, const std::string& order, int red_len) {
            return PrecisionPolicy(parse_input_precision(prec), order, red_len);
        }), py::arg("input_precision") = "ieee",
            py::arg("accumulation_order") = "k_major_sequential",
            py::arg("reduction_length") = 0)
        .def("tolerance_for", &PrecisionPolicy::tolerance_for,
             py::arg("dtype") = "f32")
        .def("multiply_accumulate", [](const PrecisionPolicy& self,
                                       py::array_t<float> a,
                                       py::array_t<float> b,
                                       py::array_t<float> acc) {
            auto a_buf = a.request();
            auto b_buf = b.request();
            auto acc_buf = acc.request();
            if (a_buf.ndim != 2 || b_buf.ndim != 2 || acc_buf.ndim != 2) {
                throw std::invalid_argument("multiply_accumulate expects 2-D tiles");
            }
            int m = static_cast<int>(a_buf.shape[0]);
            int k = static_cast<int>(a_buf.shape[1]);
            int n = static_cast<int>(b_buf.shape[1]);

            py::array_t<float> result({m, n});
            auto res_buf = result.request();
            std::memcpy(res_buf.ptr, acc_buf.ptr, m * n * sizeof(float));
            self.multiply_accumulate(
                static_cast<float*>(res_buf.ptr),
                static_cast<const float*>(a_buf.ptr), m, k,
                static_cast<const float*>(b_buf.ptr), n);
            return result;
        })
        .def("compare", [](const PrecisionPolicy& self,
                           py::array_t<float> actual,
                           py::array_t<float> reference,
                           const std::string& dtype) {
            auto a_buf = actual.request();
            auto r_buf = reference.request();
            return self.compare(
                static_cast<const float*>(a_buf.ptr),
                static_cast<const float*>(r_buf.ptr),
                a_buf.size, dtype);
        }, py::arg("actual"), py::arg("reference"), py::arg("dtype") = "f32");

    // ---- tf32_truncate ----
    m.def("tf32_truncate", [](py::array_t<float> values) {
        auto buf = values.request();
        py::array_t<float> result(buf.shape);
        auto out_buf = result.request();
        tf32_truncate(
            static_cast<float*>(out_buf.ptr),
            static_cast<const float*>(buf.ptr),
            buf.size);
        return result;
    }, "Round values to tf32 precision, round-half-to-even.");

    // ---- derive_tolerance ----
    m.def("derive_tolerance", [](const std::string& precision,
                                 int reduction_length,
                                 const std::string& dtype) {
        return derive_tolerance(
            parse_input_precision(precision), reduction_length, dtype);
    }, py::arg("precision"), py::arg("reduction_length"),
       py::arg("dtype") = "f32");

    // ---- emulate ----
    m.def("emulate", [](py::object program,
                        py::dict inputs,
                        py::object policy_obj,
                        py::tuple grid,
                        py::dict transfers) {
        // Convert program.
        Program prog = convert_program(program);
        for (auto& [key, val] : transfers) {
            std::vector<std::pair<std::string, std::string>> moves;
            for (auto& item : py::cast<py::list>(val)) {
                std::string spec = py::cast<std::string>(item);
                auto cut = spec.find('>');
                if (cut == std::string::npos) {
                    throw std::runtime_error("transfer '" + spec + "' must look like 'src>dst'");
                }
                moves.emplace_back(spec.substr(0, cut), spec.substr(cut + 1));
            }
            prog.transfers[py::cast<std::string>(key)] = std::move(moves);
        }

        // Convert inputs.
        auto cpp_inputs = convert_inputs(inputs);

        // Convert policy.
        PrecisionPolicy policy;
        if (!policy_obj.is_none()) {
            std::string prec = py::cast<std::string>(policy_obj.attr("input_precision"));
            std::string order = py::cast<std::string>(policy_obj.attr("accumulation_order"));
            int red_len = py::cast<int>(policy_obj.attr("reduction_length"));
            policy = PrecisionPolicy(parse_input_precision(prec), order, red_len);
        }

        // Grid.
        int gx = 0, gy = 0, gz = 0;
        if (grid.size() >= 1) gx = py::cast<int>(grid[0]);
        if (grid.size() >= 2) gy = py::cast<int>(grid[1]);
        if (grid.size() >= 3) gz = py::cast<int>(grid[2]);

        // Run.
        auto outputs = emulate(prog, cpp_inputs, policy, gx, gy, gz);

        return convert_outputs(outputs);
    }, py::arg("program"), py::arg("inputs"),
       py::arg("policy") = py::none(),
       py::arg("grid") = py::make_tuple(0, 0, 0),
       py::arg("transfers") = py::dict(),
       "Execute a program and return every buffer it wrote (postcondition 1).");

    // ---- HAS_CPP flag ----
    m.attr("HAS_CPP") = true;
}
