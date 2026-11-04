/**
 * @file basis_loader.cpp
 * @brief Minimal JSON reader, molecule/basis loading, and shell normalization.
 *
 * The data files are small and fully controlled by this repository, so a
 * compact recursive-descent JSON parser is sufficient and keeps the library
 * dependency free.
 */
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>

#include "cuest_internal.h"
#include "kernels/shell.cuh"

namespace cuest {
namespace {

struct JsonValue {
  enum Type { Null, Bool, Number, String, Array, Object } type = Null;
  double num = 0.0;
  bool boolean = false;
  std::string str;
  std::vector<JsonValue> arr;
  std::map<std::string, JsonValue> obj;

  const JsonValue *find(const std::string &key) const {
    auto it = obj.find(key);
    return it == obj.end() ? nullptr : &it->second;
  }
};

class JsonParser {
 public:
  explicit JsonParser(const std::string &text) : s_(text) {}

  bool parse(JsonValue &out, std::string &err) {
    skip();
    if (!value(out)) {
      err = err_;
      return false;
    }
    skip();
    if (pos_ != s_.size()) {
      err = "trailing characters in JSON";
      return false;
    }
    return true;
  }

 private:
  void skip() {
    while (pos_ < s_.size() && (std::isspace(static_cast<unsigned char>(s_[pos_])) != 0)) ++pos_;
  }

  bool value(JsonValue &out) {
    skip();
    if (pos_ >= s_.size()) return fail("unexpected end of JSON");
    const char c = s_[pos_];
    if (c == '{') return object(out);
    if (c == '[') return array(out);
    if (c == '"') {
      out.type = JsonValue::String;
      return string(out.str);
    }
    if (c == 't' || c == 'f') {
      out.type = JsonValue::Bool;
      if (s_.compare(pos_, 4, "true") == 0) {
        out.boolean = true;
        pos_ += 4;
        return true;
      }
      if (s_.compare(pos_, 5, "false") == 0) {
        out.boolean = false;
        pos_ += 5;
        return true;
      }
      return fail("invalid literal");
    }
    if (c == 'n') {
      if (s_.compare(pos_, 4, "null") == 0) {
        out.type = JsonValue::Null;
        pos_ += 4;
        return true;
      }
      return fail("invalid literal");
    }
    out.type = JsonValue::Number;
    return number(out.num);
  }

  bool number(double &v) {
    const char *start = s_.c_str() + pos_;
    char *end = nullptr;
    v = std::strtod(start, &end);
    if (end == start) return fail("invalid number");
    pos_ += static_cast<size_t>(end - start);
    return true;
  }

  bool string(std::string &out) {
    if (s_[pos_] != '"') return fail("expected string");
    ++pos_;
    out.clear();
    while (pos_ < s_.size()) {
      const char c = s_[pos_++];
      if (c == '"') return true;
      if (c == '\\') {
        if (pos_ >= s_.size()) return fail("bad escape");
        const char e = s_[pos_++];
        switch (e) {
          case '"': out.push_back('"'); break;
          case '\\': out.push_back('\\'); break;
          case '/': out.push_back('/'); break;
          case 'n': out.push_back('\n'); break;
          case 't': out.push_back('\t'); break;
          case 'r': out.push_back('\r'); break;
          default: out.push_back(e); break;
        }
      } else {
        out.push_back(c);
      }
    }
    return fail("unterminated string");
  }

  bool array(JsonValue &out) {
    out.type = JsonValue::Array;
    ++pos_;
    skip();
    if (pos_ < s_.size() && s_[pos_] == ']') {
      ++pos_;
      return true;
    }
    while (true) {
      JsonValue v;
      if (!value(v)) return false;
      out.arr.push_back(std::move(v));
      skip();
      if (pos_ >= s_.size()) return fail("unterminated array");
      if (s_[pos_] == ',') {
        ++pos_;
        continue;
      }
      if (s_[pos_] == ']') {
        ++pos_;
        return true;
      }
      return fail("expected , or ]");
    }
  }

  bool object(JsonValue &out) {
    out.type = JsonValue::Object;
    ++pos_;
    skip();
    if (pos_ < s_.size() && s_[pos_] == '}') {
      ++pos_;
      return true;
    }
    while (true) {
      skip();
      std::string key;
      if (!string(key)) return false;
      skip();
      if (pos_ >= s_.size() || s_[pos_] != ':') return fail("expected :");
      ++pos_;
      JsonValue v;
      if (!value(v)) return false;
      out.obj[key] = std::move(v);
      skip();
      if (pos_ >= s_.size()) return fail("unterminated object");
      if (s_[pos_] == ',') {
        ++pos_;
        continue;
      }
      if (s_[pos_] == '}') {
        ++pos_;
        return true;
      }
      return fail("expected , or }");
    }
  }

  bool fail(const char *msg) {
    err_ = msg;
    return false;
  }

  const std::string &s_;
  size_t pos_ = 0;
  std::string err_;
};

bool read_text(const std::string &path, std::string &text) {
  std::ifstream in(path);
  if (!in) return false;
  std::ostringstream ss;
  ss << in.rdbuf();
  text = ss.str();
  return true;
}

bool parse_json_file(const std::string &path, JsonValue &out, std::string &err) {
  std::string text;
  if (!read_text(path, text)) {
    err = "cannot open " + path;
    return false;
  }
  JsonParser p(text);
  if (!p.parse(out, err)) {
    err = path + ": " + err;
    return false;
  }
  return true;
}

}  // namespace

const char *element_symbol(int Z) {
  static const char *symbols[] = {"X",  "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",
                                  "Ne", "Na", "Mg", "Al", "Si", "P",  "S",  "Cl", "Ar"};
  if (Z < 0 || Z > 18) return "X";
  return symbols[Z];
}

bool load_molecule(const std::string &path, Molecule &mol, std::string &err) {
  JsonValue root;
  if (!parse_json_file(path, root, err)) return false;
  if (root.type != JsonValue::Object) {
    err = "molecule JSON root must be an object";
    return false;
  }
  double unit = 1.0;
  if (const JsonValue *u = root.find("units")) {
    if (u->type == JsonValue::String && u->str == "angstrom") unit = 1.8897261246257702;
  }
  const JsonValue *atoms = root.find("atoms");
  if (!atoms || atoms->type != JsonValue::Array || atoms->arr.empty()) {
    err = "molecule JSON needs a non-empty \"atoms\" array";
    return false;
  }
  mol.xyz.clear();
  mol.Z.clear();
  for (const JsonValue &a : atoms->arr) {
    if (a.type != JsonValue::Object) {
      err = "each atom must be an object";
      return false;
    }
    const JsonValue *z = a.find("Z");
    const JsonValue *xyz = a.find("xyz");
    if (!z || !xyz || xyz->type != JsonValue::Array || xyz->arr.size() != 3) {
      err = "each atom needs Z and three xyz coordinates";
      return false;
    }
    mol.Z.push_back(static_cast<int>(z->num));
    mol.xyz.push_back(xyz->arr[0].num * unit);
    mol.xyz.push_back(xyz->arr[1].num * unit);
    mol.xyz.push_back(xyz->arr[2].num * unit);
  }
  return true;
}

std::string data_directory() {
  if (const char *env = std::getenv("CUEST_DATA_DIR")) return std::string(env);
#ifdef CUEST_DATA_DIR_DEFAULT
  return std::string(CUEST_DATA_DIR_DEFAULT);
#else
  return std::string("data");
#endif
}

bool load_basis_element(const std::string &path, const std::string &element,
                        std::vector<RawShell> &shells, std::string &err) {
  JsonValue root;
  if (!parse_json_file(path, root, err)) return false;
  const JsonValue *elements = root.find("elements");
  if (!elements || elements->type != JsonValue::Object) {
    err = path + ": missing \"elements\" object";
    return false;
  }
  const JsonValue *el = elements->find(element);
  if (!el || el->type != JsonValue::Object) {
    err = path + ": basis does not contain element " + element;
    return false;
  }
  const JsonValue *sh = el->find("shells");
  if (!sh || sh->type != JsonValue::Array) {
    err = path + ": element " + element + " has no shells";
    return false;
  }
  shells.clear();
  for (const JsonValue &s : sh->arr) {
    RawShell rs;
    const JsonValue *am = s.find("am");
    const JsonValue *ex = s.find("exponents");
    const JsonValue *co = s.find("coefficients");
    if (!am || !ex || ex->type != JsonValue::Array || !co || co->type != JsonValue::Array) {
      err = path + ": malformed shell";
      return false;
    }
    rs.am = static_cast<int>(am->num);
    for (const JsonValue &e : ex->arr) rs.exps.push_back(e.num);
    for (const JsonValue &col : co->arr) {
      if (col.type != JsonValue::Array) {
        err = path + ": coefficient column must be an array";
        return false;
      }
      std::vector<double> cc;
      for (const JsonValue &c : col.arr) cc.push_back(c.num);
      if (cc.size() != rs.exps.size()) {
        err = path + ": coefficient/exponent length mismatch";
        return false;
      }
      rs.coeffs.push_back(std::move(cc));
    }
    shells.push_back(std::move(rs));
  }
  return true;
}

bool build_basis(const Molecule &mol, const std::string &basis_name_in, BasisData &out,
                 std::string &err) {
  std::string basis_name = basis_name_in;
  /* Accept both "6-31g*" and the PySCF spelling "6-31g*" (stored with a star). */
  std::string path = data_directory() + "/basis/" + basis_name + ".json";
  out = BasisData();
  out.natoms = static_cast<int>(mol.Z.size());
  out.atom_xyz = mol.xyz;
  out.atom_Z.assign(mol.Z.begin(), mol.Z.end());

  int max_am = 0;
  int func_offset = 0;
  for (int ia = 0; ia < out.natoms; ++ia) {
    const std::string element = element_symbol(mol.Z[ia]);
    std::vector<RawShell> raw;
    if (!load_basis_element(path, element, raw, err)) return false;
    for (RawShell &rs : raw) {
      if (rs.am > CUEST_MAX_AM) {
        err = "basis contains angular momentum beyond d (l > 2)";
        return false;
      }
      PrimShell shell;
      shell.atom = ia;
      shell.L = rs.am;
      shell.nprim = static_cast<int>(rs.exps.size());
      shell.nctr = static_cast<int>(rs.coeffs.size());
      shell.func_offset = func_offset;
      shell.exps = rs.exps;
      const int ncomp = shell.ncomp();
      shell.cn.assign(static_cast<size_t>(shell.nctr) * ncomp * shell.nprim, 0.0);

      /* Fold the primitive normalization N(L, alpha) into the coefficients and
         scale each Cartesian component to unit self-overlap. */
      std::vector<double> prim_norm(shell.nprim);
      for (int p = 0; p < shell.nprim; ++p) prim_norm[p] = cuest_prim_norm(rs.am, rs.exps[p]);

      for (int ictr = 0; ictr < shell.nctr; ++ictr) {
        for (int icomp = 0; icomp < ncomp; ++icomp) {
          int lx, ly, lz;
          cuest_cart_component(rs.am, icomp, &lx, &ly, &lz);
          const int comp[3] = {lx, ly, lz};
          double self = 0.0;
          for (int p = 0; p < shell.nprim; ++p) {
            for (int q = 0; q < shell.nprim; ++q) {
              const double pp = rs.exps[p] + rs.exps[q];
              double mom = 1.0;
              for (int d = 0; d < 3; ++d) mom *= cuest_gauss_moment_1d(2 * comp[d], pp);
              self += rs.coeffs[ictr][p] * prim_norm[p] * rs.coeffs[ictr][q] * prim_norm[q] * mom;
            }
          }
          if (!(self > 0.0) || !std::isfinite(self)) {
            err = "basis normalization failed (non-positive self overlap)";
            return false;
          }
          const double scale = 1.0 / std::sqrt(self);
          double *dst = &shell.cn[(static_cast<size_t>(ictr) * ncomp + icomp) * shell.nprim];
          for (int p = 0; p < shell.nprim; ++p)
            dst[p] = rs.coeffs[ictr][p] * prim_norm[p] * scale;
        }
      }
      func_offset += shell.nctr * ncomp;
      if (rs.am > max_am) max_am = rs.am;
      out.shells.push_back(std::move(shell));
    }
  }
  out.nshells = static_cast<int>(out.shells.size());
  out.nbf = func_offset;
  out.max_am = max_am;
  if (out.nbf == 0) {
    err = "basis produced zero functions";
    return false;
  }
  return true;
}

}  // namespace cuest
