#pragma once

#include "Graph.h"
#include "UnitHeap.h"
#include "Util.h"
#include <iostream>


std::tuple<std::vector<int>, std::vector<int>, std::vector<int>>
gorder_reorder(const std::vector<int> &src, const std::vector<int> &dst, int W=5);

