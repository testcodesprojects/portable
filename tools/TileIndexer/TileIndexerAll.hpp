/**
 * @file    TileIndexerAll.hpp
 * @brief   Convenience umbrella header aggregating public TileIndexer
 *          components for tools, tests, and admin-side benchmarks.
 *
 * @project sTiles (Sparse Tiles Library)
 * @author  Esmail Abdul Fattah, King Abdullah University of Science and Technology (KAUST)
 * @contact esmail.abdulfattah@kaust.edu.sa
 * @version 3.0.0
 * @date 1 1 2026
 * @license Proprietary
 *
 * @note This file is part of the sTiles library, a proprietary software package.
 *       Redistribution or modification without prior permission is prohibited.
 *
 * Copyright (c) 2026, Esmail Abdul Fattah, KAUST. All rights reserved.
 *
 * @license
 * This software is proprietary and confidential. Unauthorized copying, distribution, or modification
 * of this software, via any medium, is strictly prohibited. Permission is granted to use the software
 * in binary form for non-commercial purposes only, provided that this copyright notice and permission
 * notice are included in all copies or substantial portions of the software.
 *
 * DISCLAIMER:
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
 * NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM, OUT OF, OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#pragma once

// Central include aggregating the public TileIndexer components required by
// the admin-side benchmarks and tests.  Pulling them together in one header
// keeps downstream sources tidy and ensures a consistent include order.

#include "TileIndexer.hpp"
#include "TileIndexerCounter.hpp"
#include "MatrixIO.hpp"
#include "TileIndexerFill.hpp"
#include "TileIndexerGraphBuilder.hpp"
#include "TileIndexerMapper.hpp"
#include "TileIndexerPrinter.hpp"
#include "TileIndexerAutoStrict.hpp"
#include "TileIndexerMemoryUtils.hpp"
