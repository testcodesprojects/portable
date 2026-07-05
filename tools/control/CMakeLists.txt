# ------------------------------
# control/CMakeLists.txt
# ------------------------------
#
set(SRC
    allocate.c async.c auxiliary.c
    context.c control.c descriptor.cpp
    stilesos.c stiles_threadsetting.c
)
add_library(control OBJECT ${SRC})
target_include_directories(control PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)
foreach(t stiles_static stiles_shared)
    if(TARGET ${t})
        target_sources(${t} PRIVATE $<TARGET_OBJECTS:control>)
    endif()
endforeach()

