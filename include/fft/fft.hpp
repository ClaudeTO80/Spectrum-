#pragma once

// Basic API: Spectrum::forward, Spectrum::inverse.
#include "simple.hpp"

// Filtering: low-pass, high-pass, band-pass and band-stop of real signals.
#include "filter.hpp"

// Advanced API: explicit plans, normalization, algorithm selection, workspaces.
#include "exceptions.hpp"
#include "fft_plan.hpp"
#include "nd_plan.hpp"
#include "real_plan.hpp"
#include "shape.hpp"
#include "types.hpp"
#include "version.hpp"
#include "workspace.hpp"
