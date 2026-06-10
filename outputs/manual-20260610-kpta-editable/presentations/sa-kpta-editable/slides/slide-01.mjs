const C = {
  ink: "#172033",
  muted: "#526078",
  line: "#AAB6C8",
  blue: "#2463C7",
  blueLight: "#EAF2FF",
  orange: "#F07A24",
  orangeLight: "#FFF1E6",
  purple: "#7A4CC2",
  purpleLight: "#F2ECFF",
  teal: "#168B91",
  tealLight: "#E8F7F6",
  red: "#D94747",
  redLight: "#FFF0F0",
  grayLight: "#F5F7FA",
  white: "#FFFFFF",
};

function box(ctx, slide, x, y, w, h, fill, stroke, radius = false, width = 2) {
  return ctx.addShape(slide, {
    x, y, width: w, height: h,
    geometry: radius ? "roundRect" : "rect",
    fill,
    line: ctx.line(stroke, width),
  });
}

function text(ctx, slide, value, x, y, w, h, size = 18, color = C.ink, bold = false, align = "center") {
  return ctx.addText(slide, {
    text: value, x, y, width: w, height: h,
    fontSize: size, color, bold, align, valign: "middle",
    insets: { left: 4, right: 4, top: 2, bottom: 2 },
  });
}

function line(ctx, slide, x1, y1, x2, y2, color = C.ink, width = 3, arrow = true) {
  const horizontal = Math.abs(y2 - y1) <= Math.abs(x2 - x1);
  if (horizontal) {
    const x = Math.min(x1, x2);
    const w = Math.max(2, Math.abs(x2 - x1) - (arrow ? 10 : 0));
    ctx.addShape(slide, { x, y: y1 - width / 2, width: w, height: width, geometry: "rect", fill: color, line: ctx.line(color, 0) });
    if (arrow) text(ctx, slide, x2 >= x1 ? "▶" : "◀", x2 >= x1 ? x2 - 13 : x2, y2 - 10, 14, 20, 14, color, true);
  } else {
    const y = Math.min(y1, y2);
    const h = Math.max(2, Math.abs(y2 - y1) - (arrow ? 10 : 0));
    ctx.addShape(slide, { x: x1 - width / 2, y, width, height: h, geometry: "rect", fill: color, line: ctx.line(color, 0) });
    if (arrow) text(ctx, slide, y2 >= y1 ? "▼" : "▲", x2 - 9, y2 >= y1 ? y2 - 15 : y2, 18, 18, 13, color, true);
  }
}

function featureStack(ctx, slide, x, y, count, color, scale = 1) {
  for (let i = count - 1; i >= 0; i -= 1) {
    box(ctx, slide, x + i * 5 * scale, y - i * 4 * scale, 38 * scale, 54 * scale, color, C.white, false, 1);
  }
}

function section(ctx, slide, titleValue, x, y, w, h, color) {
  box(ctx, slide, x, y, w, h, C.white, color, true, 2);
  text(ctx, slide, titleValue, x + 10, y - 16, w - 20, 30, 17, color, true);
}

function stage(ctx, slide, label, x, y, w, h, fill, stroke) {
  box(ctx, slide, x, y, w, h, fill, stroke, true, 1.5);
  text(ctx, slide, label, x, y, w, h, 14, C.ink, true);
}

export default async function addSlide(presentation, ctx) {
  const slide = presentation.slides.add();
  slide.background.fill = C.white;

  text(ctx, slide, "SA-KPTA-Net: Slice-Aware 2.5D Kinetic Prior-Guided Temporal Attention Network",
    30, 15, 1540, 48, 31, C.ink, true, "left");
  text(ctx, slide, "Scheme D · Fully editable architecture diagram", 1160, 24, 390, 28, 15, C.muted, false, "right");

  // Input
  section(ctx, slide, "MULTI-PHASE 2.5D INPUT", 20, 95, 185, 615, C.ink);
  text(ctx, slide, "Three neighboring slices", 34, 118, 157, 28, 17, C.ink, true);
  const sliceYs = [170, 335, 500];
  const sliceNames = ["z − 1", "z", "z + 1"];
  sliceYs.forEach((sy, idx) => {
    text(ctx, slide, sliceNames[idx], 28, sy + 34, 42, 28, 18, C.ink, true);
    box(ctx, slide, 72, sy, 92, 92, "#DCE3EB", C.ink, true, 1.5);
    ctx.addShape(slide, { x: 89, y: sy + 17, width: 58, height: 58, geometry: "ellipse", fill: "#68778C", line: ctx.line("#68778C", 0) });
    ctx.addShape(slide, { x: 105, y: sy + 27, width: 27, height: 38, geometry: "ellipse", fill: "#E8EDF3", line: ctx.line("#E8EDF3", 0) });
    ctx.addShape(slide, { x: 125, y: sy + 36, width: 10, height: 13, geometry: "ellipse", fill: C.red, line: ctx.line(C.red, 0) });
    for (let p = 0; p < 5; p += 1) {
      box(ctx, slide, 152 + p * 5, sy + 17 + p * 4, 18, 56, p < 2 ? C.blueLight : C.orangeLight, p < 2 ? C.blue : C.orange, false, 1);
    }
  });
  text(ctx, slide, "Pre · Post1–8 · SUB1–8", 28, 625, 165, 30, 14, C.muted, false);
  text(ctx, slide, "X ∈ ℝ B×3×17×H×W", 28, 663, 165, 28, 16, C.ink, true);

  // Appearance stream
  section(ctx, slide, "SLICE-AWARE SPATIOTEMPORAL REPRESENTATION", 225, 95, 510, 310, C.blue);
  box(ctx, slide, 242, 130, 132, 235, C.blueLight, C.blue, true, 1.5);
  text(ctx, slide, "Shared Slice-wise\nCNN Stem", 250, 142, 116, 52, 18, C.blue, true);
  featureStack(ctx, slide, 280, 215, 4, "#76A5EA", 1);
  text(ctx, slide, "independent for every\nslice and phase", 248, 285, 120, 45, 13, C.muted);
  text(ctx, slide, "[B,3,17,C,H,W]", 248, 333, 120, 22, 13, C.ink, true);

  line(ctx, slide, 375, 246, 397, 246, C.blue, 3, true);
  box(ctx, slide, 400, 130, 140, 235, C.blueLight, C.blue, true, 1.5);
  text(ctx, slide, "CSAM Slice Context\nAggregation", 408, 142, 124, 52, 18, C.blue, true);
  ["semantic", "spatial", "inter-slice"].forEach((v, i) => {
    stage(ctx, slide, v, 420, 210 + i * 35, 100, 25, C.white, C.blue);
  });
  text(ctx, slide, "aggregate z−1, z, z+1\npreserve T=17 phases", 408, 312, 124, 35, 12, C.muted);
  text(ctx, slide, "[B,17,C,H,W]", 408, 343, 124, 18, 13, C.ink, true);

  line(ctx, slide, 541, 246, 558, 246, C.blue, 3, true);
  box(ctx, slide, 560, 118, 160, 265, C.blueLight, C.blue, true, 2);
  text(ctx, slide, "PDWA", 570, 130, 140, 30, 21, C.blue, true);
  text(ctx, slide, "Phase-Difference\nWeighting Attention", 570, 158, 140, 45, 15, C.ink, true);
  stage(ctx, slide, "feature score", 575, 216, 125, 26, C.white, C.blue);
  stage(ctx, slide, "SUB enhancement prior", 575, 248, 125, 26, C.white, C.orange);
  stage(ctx, slide, "kinetic bias", 575, 280, 125, 26, C.orangeLight, C.orange);
  text(ctx, slide, "softmax over 17 phases", 572, 311, 136, 22, 12, C.muted);
  const heat = ["#153C9B", "#2463C7", "#1AA6C8", "#48C78E", "#F0C419", "#F07A24", "#D94747", "#F07A24", "#2463C7"];
  heat.forEach((c, i) => box(ctx, slide, 578 + i * 14, 338, 14, 16, c, C.white, false, 0.5));
  text(ctx, slide, "αt", 704, 335, 14, 20, 12, C.ink, true);
  text(ctx, slide, "Fphase", 610, 358, 60, 20, 14, C.ink, true);

  // Kinetic stream
  section(ctx, slide, "KINETIC PRIOR MODELING", 225, 430, 510, 245, C.orange);
  box(ctx, slide, 242, 465, 265, 175, C.orangeLight, C.orange, true, 1.5);
  text(ctx, slide, "Pseudo-Kinetic Map Builder", 250, 476, 249, 28, 18, C.orange, true);
  const maps = [
    ["Peak", "#1A5FB4"], ["Mean", "#18A6A6"], ["Temporal\nSTD", "#57C785"],
    ["Early", "#F2C94C"], ["Late", "#F07A24"],
  ];
  maps.forEach(([lab, col], i) => {
    box(ctx, slide, 253 + i * 48, 525, 38, 54, col, C.white, true, 1);
    ctx.addShape(slide, { x: 264 + i * 48, y: 540, width: 16, height: 20, geometry: "ellipse", fill: C.redLight, line: ctx.line(C.white, 1) });
    text(ctx, slide, lab, 249 + i * 48, 584, 46, 30, 11, C.ink, true);
  });
  text(ctx, slide, "K ∈ ℝ B×3×5×H×W", 290, 615, 170, 18, 13, C.ink, true);

  line(ctx, slide, 508, 550, 526, 550, C.orange, 3, true);
  box(ctx, slide, 528, 465, 185, 175, C.orangeLight, C.orange, true, 1.5);
  text(ctx, slide, "Kinetic Prior Encoder\n+ Slice Aggregation", 538, 482, 165, 48, 18, C.orange, true);
  featureStack(ctx, slide, 585, 548, 4, "#F5A565", 0.9);
  text(ctx, slide, "Fkin ∈ ℝ B×C×H×W", 540, 612, 160, 22, 13, C.ink, true);

  // Dual guidance arrows
  line(ctx, slide, 638, 465, 638, 395, C.orange, 3, true);
  text(ctx, slide, "kinetic bias +\ndifference refinement", 642, 399, 98, 42, 12, C.orange, true, "left");

  // Fusion
  section(ctx, slide, "DUAL-ROLE FUSION", 755, 215, 125, 260, C.orange);
  box(ctx, slide, 768, 255, 99, 175, C.orangeLight, C.orange, true, 2);
  text(ctx, slide, "Kinetic–Raw", 774, 268, 87, 22, 14, C.orange, true);
  text(ctx, slide, "Residual Fusion", 774, 291, 87, 22, 14, C.orange, true);
  text(ctx, slide, "Fphase + Fkin", 776, 330, 83, 22, 12, C.ink, true);
  ctx.addShape(slide, { x: 804, y: 375, width: 30, height: 30, geometry: "ellipse", fill: C.white, line: ctx.line(C.orange, 2) });
  text(ctx, slide, "+", 804, 374, 30, 30, 22, C.orange, true);
  text(ctx, slide, "F0", 800, 409, 40, 20, 14, C.ink, true);
  line(ctx, slide, 714, 550, 742, 550, C.orange, 3, false);
  line(ctx, slide, 742, 550, 742, 390, C.orange, 3, false);
  line(ctx, slide, 742, 390, 768, 390, C.orange, 3, true);
  text(ctx, slide, "explicit kinetic feature", 655, 523, 86, 25, 11, C.orange, true);
  line(ctx, slide, 720, 250, 768, 300, C.blue, 3, true);

  // Encoder / decoder
  section(ctx, slide, "HYBRID MULTISCALE ENCODING", 900, 95, 350, 580, C.purple);
  text(ctx, slide, "CNN Encoder", 916, 122, 145, 25, 17, C.blue, true);
  text(ctx, slide, "U-Net Decoder", 1085, 122, 145, 25, 17, C.teal, true);
  const enc = [
    ["C · H", 158, 130], ["2C · H/2", 225, 120], ["4C · H/4", 292, 110],
    ["8C · H/8", 359, 100], ["16C · H/16", 426, 90],
  ];
  enc.forEach(([lab, yy, ww], i) => {
    stage(ctx, slide, lab, 925, yy, ww, 38, i === 4 ? C.purpleLight : C.blueLight, i === 4 ? C.purple : C.blue);
    if (i < 4) line(ctx, slide, 970, yy + 39, 970, yy + 65, C.blue, 2, true);
  });
  box(ctx, slide, 915, 510, 150, 70, C.purpleLight, C.purple, true, 2);
  text(ctx, slide, "Swin Transformer Bottleneck", 922, 519, 136, 25, 15, C.purple, true);
  text(ctx, slide, "depth=2 · window=7 · heads=4", 922, 547, 136, 20, 12, C.ink, true);
  line(ctx, slide, 970, 464, 970, 510, C.purple, 2, true);

  const decY = [426, 359, 292, 225];
  const decLab = ["8C", "4C", "2C", "C"];
  decY.forEach((yy, i) => {
    stage(ctx, slide, decLab[i], 1120, yy, 78, 38, C.tealLight, C.teal);
    if (i < 3) line(ctx, slide, 1159, yy, 1159, decY[i + 1] + 38, C.teal, 2, true);
  });
  [0, 1, 2, 3].forEach((i) => {
    const sy = enc[i][1] + 19;
    const ty = decY[3 - i] + 19;
    ctx.addShape(slide, { x: 1058, y: sy - 1, width: 60, height: 2, geometry: "rect", fill: C.line, line: ctx.line(C.line, 0) });
    text(ctx, slide, "···", 1070, sy - 11, 35, 20, 16, C.muted, true);
    ctx.addShape(slide, { x: 1112, y: Math.min(sy, ty), width: 2, height: Math.max(2, Math.abs(ty - sy)), geometry: "rect", fill: C.line, line: ctx.line(C.line, 0) });
  });
  line(ctx, slide, 880, 390, 912, 390, C.ink, 3, true);

  // Refinement
  section(ctx, slide, "UNCERTAINTY-AWARE REFINEMENT", 1270, 95, 310, 580, C.red);
  box(ctx, slide, 1290, 132, 270, 66, C.grayLight, C.ink, true, 1.5);
  text(ctx, slide, "Coarse Segmentation Head", 1300, 142, 250, 24, 17, C.ink, true);
  text(ctx, slide, "Pc = σ(Conv1×1(Fdec))", 1300, 168, 250, 20, 13, C.muted);
  line(ctx, slide, 1425, 199, 1425, 220, C.red, 2, true);
  box(ctx, slide, 1290, 224, 270, 72, C.redLight, C.red, true, 2);
  text(ctx, slide, "Uncertainty Map", 1300, 232, 250, 25, 18, C.red, true);
  text(ctx, slide, "U = 1 − |2Pc − 1|", 1300, 260, 250, 25, 17, C.ink, true);
  line(ctx, slide, 1425, 297, 1425, 321, C.red, 2, true);
  box(ctx, slide, 1290, 325, 270, 105, C.redLight, C.red, true, 2);
  text(ctx, slide, "Uncertainty-Guided\nBoundary Refinement", 1300, 338, 250, 48, 19, C.red, true);
  text(ctx, slide, "decoder feature + uncertainty gating", 1300, 390, 250, 22, 12, C.muted);
  line(ctx, slide, 1199, 244, 1278, 377, C.teal, 3, true);

  const out = [
    ["Tumor\nSeg.\n(final)", C.blue, C.blueLight],
    ["Boundary\nPred.\n(aux.)", C.red, C.redLight],
    ["Uncertainty\nSupervision\n(aux.)", C.orange, C.orangeLight],
  ];
  out.forEach(([lab, stroke, fill], i) => {
    const xx = 1290 + i * 92;
    line(ctx, slide, 1336 + i * 92, 431, 1336 + i * 92, 460, stroke, 2, true);
    box(ctx, slide, xx, 465, 86, 145, fill, stroke, true, 1.5);
    text(ctx, slide, lab, xx + 4, 470, 78, 55, 12, stroke, true);
    ctx.addShape(slide, { x: xx + 23, y: 527, width: 40, height: 55, geometry: "ellipse", fill: i === 0 ? C.blue : C.white, line: ctx.line(stroke, i === 0 ? 1 : 3) });
    text(ctx, slide, i === 0 ? "Ŷ" : i === 1 ? "B̂" : "Û", xx + 25, 586, 36, 18, 13, C.ink, true);
  });

  // Loss and legend
  box(ctx, slide, 225, 725, 1030, 125, C.grayLight, C.line, true, 1.5);
  text(ctx, slide, "Overall training objective", 245, 738, 250, 26, 17, C.ink, true, "left");
  text(ctx, slide,
    "L = LDice+BCE + λb Lboundary + λu Luncertainty + λa Lattention-smooth",
    255, 770, 960, 42, 23, C.ink, true);
  const legend = [
    [C.blue, "appearance feature"], [C.orange, "kinetic prior / guidance"],
    [C.purple, "transformer bottleneck"], [C.teal, "decoder feature"], [C.red, "auxiliary supervision"],
  ];
  legend.forEach(([color, label], i) => {
    box(ctx, slide, 260 + i * 185, 818, 18, 18, color, color, true, 0);
    text(ctx, slide, label, 283 + i * 185, 813, 155, 28, 12, C.muted, false, "left");
  });

  box(ctx, slide, 1270, 725, 310, 125, C.white, C.line, true, 1.5);
  text(ctx, slide, "Key contribution", 1285, 738, 280, 24, 17, C.ink, true, "left");
  text(ctx, slide,
    "Kinetic priors guide phase attention and explicitly enter the encoder through residual fusion.",
    1285, 770, 280, 62, 15, C.muted, false, "left");

  return slide;
}
