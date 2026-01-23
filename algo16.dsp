import("stdfaust.lib");

replace = !,_;
process = ["freq":replace, "gain":replace, "gate":replace -> dx.algorithm(16)] <: _;
