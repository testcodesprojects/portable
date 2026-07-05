# ------------------------------
# compute/CMakeLists.txt
# ------------------------------
#
set(SRC
    dpotrf.cpp pdpotrf.cpp dpotri.cpp dlogdet.cpp
    dtrsm.cpp pdtrsm.cpp pdtrtri.cpp
)
add_library(compute OBJECT ${SRC})
target_include_directories(compute PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)
foreach(t stiles_static stiles_shared)
    if(TARGET ${t})
        target_sources(${t} PRIVATE $<TARGET_OBJECTS:compute>)
    endif()
endforeach()

