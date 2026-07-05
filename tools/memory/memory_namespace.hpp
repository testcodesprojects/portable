/**
 * @file memory_namespace.hpp
 * @brief Unified namespace for all sTiles memory managers.
 *
 * Provides a single access point to all memory management classes under
 * the sTiles::Memory namespace for cleaner code organization.
 *
 * @author Esmail Abdul Fattah
 * @affiliation King Abdullah University of Science and Technology (KAUST)
 * @version 3.0.0
 *
 * @copyright Copyright (c) 2026, KAUST. All rights reserved.
 *
 * PROPRIETARY AND CONFIDENTIAL
 * This software is the exclusive property of KAUST. Unauthorized copying, modification,
 * distribution, or use of this software, in whole or in part, is strictly prohibited.
 * This code may not be reproduced, reverse-engineered, or used to create derivative works
 * without explicit written permission from the copyright holder.
 */

#pragma once

// Unify access to all memory managers under sTiles::Memory::

// Core/general managers
#include "MemoryManager.hpp"
#include "TileMemoryManager.hpp"
#include "TreeMemoryManager.hpp"
#include "AlgorithmsMemoryManager.hpp"
#include "OrderingMemoryManager.hpp"
#include "TileIndexerMemoryManager.hpp"

// CPU manager now lives within tools/memory
#include "cpuSmartTileMemoryManager.hpp"
#ifdef STILES_GPU
#include "../gpu/GpuMemoryManager.hpp"
#endif

namespace sTiles { namespace Memory {

using MemoryManager            = ::MemoryManager;
using TileMemoryManager        = ::TileMemoryManager;
using TreeMemoryManager        = ::TreeMemoryManager;
using AlgorithmsMemoryManager  = ::AlgorithmsMemoryManager;
using OrderingMemoryManager    = ::OrderingMemoryManager;
using TileIndexerMemoryManager = sTiles::TileIndexerMemoryManager;
using CpuMemoryManager         = sTiles::CpuMemoryManager;
// Preferred descriptive alias for SmartTile CPU allocations
using CpuSmartTileMemoryManager = sTiles::CpuMemoryManager;

#ifdef STILES_GPU
using GpuMemoryManager         = ::GpuMemoryManager;
#endif

} } // namespace sTiles::Memory
