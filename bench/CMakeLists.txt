add_executable(bench_integrals bench_integrals.cpp)
target_link_libraries(bench_integrals PRIVATE cuest_eri)
target_include_directories(bench_integrals PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_options(bench_integrals PRIVATE -Wall -Wextra)
