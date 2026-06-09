import common from "./common.wgsl?raw";
import bindSim from "./bindSim.wgsl?raw";
import bindA from "./bindA.wgsl?raw";
import bindB from "./bindB.wgsl?raw";
import bindBCollide from "./bindBCollide.wgsl?raw";
import bindStat from "./bindStat.wgsl?raw";

import integrate from "./integrate.wgsl?raw";
import apply from "./apply.wgsl?raw";
import clearA from "./clearA.wgsl?raw";
import binA from "./binA.wgsl?raw";
import collideA from "./collideA.wgsl?raw";
import clearB from "./clearB.wgsl?raw";
import binB from "./binB.wgsl?raw";
import fixB from "./fixB.wgsl?raw";
import collideB from "./collideB.wgsl?raw";
import render from "./render.wgsl?raw";
import clearStat from "./clearStat.wgsl?raw";
import binStat from "./binStat.wgsl?raw";
import overlay from "./overlay.wgsl?raw";

export const MAX_PER_CELL = 16;
export const WG = 64;

const link = (...parts: string[]) => parts.join("\n");

export const INTEGRATE_WGSL = link(common, bindSim, integrate);
export const APPLY_WGSL = link(common, bindSim, apply);

export const CLEAR_A_WGSL = link(common, bindSim, bindA, clearA);
export const BIN_A_WGSL = link(common, bindSim, bindA, binA);
export const COLLIDE_A_WGSL = link(common, bindSim, bindA, collideA);

export const CLEAR_B_WGSL = link(common, bindSim, bindB, clearB);
export const BIN_B_WGSL = link(common, bindSim, bindB, binB);
export const FIX_B_WGSL = link(common, bindSim, bindB, fixB);
export const COLLIDE_B_WGSL = link(common, bindSim, bindBCollide, collideB);

export const RENDER_WGSL = link(common, render);

export const CLEAR_STAT_WGSL = link(common, bindStat, clearStat);
export const BIN_STAT_WGSL = link(common, bindStat, binStat);
export const OVERLAY_WGSL = link(common, overlay);
