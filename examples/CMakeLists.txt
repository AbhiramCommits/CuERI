add_executable(water_fock water_fock.c)
target_link_libraries(water_fock PRIVATE cuest_eri)
target_include_directories(water_fock PRIVATE ${CMAKE_SOURCE_DIR}/include)
set_target_properties(water_fock PROPERTIES C_STANDARD 99)
