## (2025.06.02) Trial 005: resume ifdef wrappers and add control in setup:

Comparing to trial 004, resume the `#ifdef GXL_GEDLIB_SHARED` and `#ifndef SRC_ENV_GED_ENV_GXL_CPP_`
wrappers in `gedlib/ged_env.hpp`, and add `"-DGXL_GEDLIB_SHARED"` in `gedlibpy/setup_core.py` -> `get_extensions()->Extension.extra_compile_args`.

**This does not solve the multiple definition error.**

## (2025.06.02) Trial 004: remove `"include/gedlib-master/src/env/ged_env.gxl.cpp"`:

Comparing to trial 003, remove `"include/gedlib-master/src/env/ged_env.gxl.cpp"` from
`gedlibpy/setup_core.py` -> `get_extensions()->Extension.sources`.

**This does not solve the multiple definition error.**

## (2025.06.02) Trial 003: remove ifdef wrappers:

Comparing to trial 002, remove the `#ifdef GXL_GEDLIB_SHARED` and `#ifndef SRC_ENV_GED_ENV_GXL_CPP_` 
wrappers in `gedlib/ged_env.hpp`.

**This does not solve the multiple definition error.**

## (2025.06.02) Trial 002: Add `#define SRC_ENV_GED_ENV_GXL_CPP_`:

Comparing to trial 001, add `#define SRC_ENV_GED_ENV_GXL_CPP_` in `gedlibpy/src/gedlib_header.hpp`.

**This does not solve the multiple definition error.**

## (2025.06.02) Trial 001: Add extern declaration and instantiation for `AttrLabel` 

In `gedlib/ged_env.hpp`, add `extern template class GEDEnv<GXLNodeID, ged::AttrLabel, GXLLabel>;` as
follows:

```cpp
#ifdef GXL_GEDLIB_SHARED
#ifndef SRC_ENV_GED_ENV_GXL_CPP_
extern template class GEDEnv<GXLNodeID, GXLLabel, GXLLabel>;
extern template class GEDEnv<GXLNodeID, ged::AttrLabel, GXLLabel>;
#endif /* SRC_ENV_GED_ENV_GXL_CPP_ */
#endif /* GXL_GEDLIB_SHARED */
```

Add `template class GEDEnv<GXLNodeID, ged::AttrLabel, ged::AttrLabel>;` in `ged_env.gxl.cpp`.

Add `"include/gedlib-master/src/env/ged_env.gxl.cpp"` in 
`gedlibpy/setup_core.py` as 

```python
def get_extensions():
	exts = [
		Extension(
			"gedlibpy",
			# sources=["gedlibpy.pyx", "src/GedLibBind.cpp"],
			sources=[
				"common_bind.pyx", "gedlibpy_gxl.pyx", "gedlibpy_attr.pyx",
				"src/gedlib_bind_gxl.cpp", "src/gedlib_bind_attr.cpp", "src/gedlib_bind_util.cpp",
				"include/gedlib-master/src/env/ged_env.gxl.cpp"
			],
.....
```

**This does not solve the multiple definition error.** E.g.,
```bash
graphkit-learn/gklearn/gedlib/src/../include/gedlib-master/src/env/ged_env.ipp:255: multiple definition of `ged::GEDEnv<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > > > >, std::map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::less<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > > > > >::load_gxl_graphs(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, ged::Options::GXLNodeEdgeType, ged::Options::GXLNodeEdgeType, std::unordered_set<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::hash<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::equal_to<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > > > const&, std::unordered_set<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::hash<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::equal_to<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > > > const&)'; build/temp.linux-x86_64-cpython-310/common_bind.o:/mnt/F0CA2D75CA2D38EC/research-repo/codes/graphkit-learn/gklearn/gedlib/./src/../include/gedlib-master/src/env/ged_env.ipp:255: first defined here
```
