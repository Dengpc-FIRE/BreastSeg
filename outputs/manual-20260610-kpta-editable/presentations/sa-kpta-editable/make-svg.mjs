import fs from "node:fs";
import path from "node:path";

const out = process.argv[2];
if (!out) throw new Error("Usage: node make-svg.mjs <output.svg>");

const esc = (s) => String(s).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
const parts = [];
const rect = (x,y,w,h,fill,stroke="#172033",rx=10,sw=2) =>
  parts.push(`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${rx}" fill="${fill}" stroke="${stroke}" stroke-width="${sw}"/>`);
const txt = (s,x,y,size=16,color="#172033",weight=400,anchor="middle") =>
  parts.push(`<text x="${x}" y="${y}" font-family="Arial, Helvetica, sans-serif" font-size="${size}" fill="${color}" font-weight="${weight}" text-anchor="${anchor}">${esc(s)}</text>`);
const arrow = (x1,y1,x2,y2,color="#172033",sw=3) =>
  parts.push(`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${sw}" marker-end="url(#arrow-${color.slice(1)})"/>`);
const multiline = (lines,x,y,size,color="#172033",weight=400,step=20) => {
  parts.push(`<text x="${x}" y="${y}" font-family="Arial, Helvetica, sans-serif" font-size="${size}" fill="${color}" font-weight="${weight}" text-anchor="middle">`);
  lines.forEach((line,i) => parts.push(`<tspan x="${x}" dy="${i===0?0:step}">${esc(line)}</tspan>`));
  parts.push(`</text>`);
};

const colors = ["172033","2463C7","F07A24","7A4CC2","168B91","D94747","AAB6C8"];
parts.push(`<?xml version="1.0" encoding="UTF-8"?>`);
parts.push(`<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="900" viewBox="0 0 1600 900">`);
parts.push(`<defs>`);
colors.forEach(c => parts.push(`<marker id="arrow-${c}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#${c}"/></marker>`));
parts.push(`</defs><rect width="1600" height="900" fill="#FFFFFF"/>`);
txt("SA-KPTA-Net: Slice-Aware 2.5D Kinetic Prior-Guided Temporal Attention Network", 34, 48, 31, "#172033", 700, "start");
txt("Scheme D · Fully editable vector architecture", 1560, 46, 15, "#526078", 400, "end");

rect(20,95,185,615,"#FFFFFF","#172033",14,2);
txt("MULTI-PHASE 2.5D INPUT",112,88,17,"#172033",700);
txt("Three neighboring slices",112,132,17,"#172033",700);
[["z − 1",170],["z",335],["z + 1",500]].forEach(([lab,y])=>{
  txt(lab,48,y+55,18,"#172033",700);
  rect(72,y,92,92,"#DCE3EB","#172033",8,1.5);
  parts.push(`<ellipse cx="118" cy="${y+46}" rx="29" ry="29" fill="#68778C"/>`);
  parts.push(`<ellipse cx="118" cy="${y+46}" rx="14" ry="20" fill="#E8EDF3"/>`);
  parts.push(`<ellipse cx="130" cy="${y+43}" rx="5" ry="7" fill="#D94747"/>`);
  for(let p=0;p<5;p++) rect(152+p*5,y+17+p*4,18,56,p<2?"#EAF2FF":"#FFF1E6",p<2?"#2463C7":"#F07A24",1,1);
});
txt("Pre · Post1–8 · SUB1–8",112,650,14,"#526078");
txt("X ∈ ℝ B×3×17×H×W",112,686,16,"#172033",700);

rect(225,95,510,310,"#FFFFFF","#2463C7",14,2);
txt("SLICE-AWARE SPATIOTEMPORAL REPRESENTATION",480,88,17,"#2463C7",700);
rect(242,130,132,235,"#EAF2FF","#2463C7",12,1.5);
multiline(["Shared Slice-wise","CNN Stem"],308,162,18,"#2463C7",700,22);
for(let i=3;i>=0;i--) rect(280+i*5,215-i*4,38,54,"#76A5EA","#FFFFFF",1,1);
multiline(["independent for every","slice and phase"],308,302,13,"#526078",400,17);
txt("[B,3,17,C,H,W]",308,348,13,"#172033",700);
arrow(375,246,397,246,"#2463C7");
rect(400,130,140,235,"#EAF2FF","#2463C7",12,1.5);
multiline(["CSAM Slice Context","Aggregation"],470,162,18,"#2463C7",700,22);
["semantic","spatial","inter-slice"].forEach((s,i)=>{rect(420,210+i*35,100,25,"#FFFFFF","#2463C7",6,1);txt(s,470,228+i*35,12,"#172033",600);});
multiline(["aggregate z−1, z, z+1","preserve T=17 phases"],470,324,12,"#526078",400,16);
txt("[B,17,C,H,W]",470,351,13,"#172033",700);
arrow(541,246,558,246,"#2463C7");
rect(560,118,160,265,"#EAF2FF","#2463C7",12,2);
txt("PDWA",640,151,21,"#2463C7",700);
multiline(["Phase-Difference","Weighting Attention"],640,180,15,"#172033",700,19);
[["feature score","#2463C7"],["SUB enhancement prior","#F07A24"],["kinetic bias","#F07A24"]].forEach(([s,c],i)=>{rect(575,216+i*32,125,26,i===2?"#FFF1E6":"#FFFFFF",c,6,1);txt(s,637,234+i*32,12,"#172033",600);});
txt("softmax over 17 phases",640,326,12,"#526078");
["#153C9B","#2463C7","#1AA6C8","#48C78E","#F0C419","#F07A24","#D94747","#F07A24","#2463C7"].forEach((c,i)=>rect(578+i*14,338,14,16,c,"#FFFFFF",0,0.5));
txt("Fphase",640,375,14,"#172033",700);

rect(225,430,510,245,"#FFFFFF","#F07A24",14,2);
txt("KINETIC PRIOR MODELING",480,423,17,"#F07A24",700);
rect(242,465,265,175,"#FFF1E6","#F07A24",12,1.5);
txt("Pseudo-Kinetic Map Builder",374,496,18,"#F07A24",700);
[["Peak","#1A5FB4"],["Mean","#18A6A6"],["Temporal STD","#57C785"],["Early","#F2C94C"],["Late","#F07A24"]].forEach(([s,c],i)=>{rect(253+i*48,525,38,54,c,"#FFFFFF",6,1);parts.push(`<ellipse cx="${272+i*48}" cy="552" rx="8" ry="10" fill="#FFF0F0"/>`);txt(s,272+i*48,600,10,"#172033",700);});
txt("K ∈ ℝ B×3×5×H×W",374,629,13,"#172033",700);
arrow(508,550,526,550,"#F07A24");
rect(528,465,185,175,"#FFF1E6","#F07A24",12,1.5);
multiline(["Kinetic Prior Encoder","+ Slice Aggregation"],620,500,18,"#F07A24",700,23);
for(let i=3;i>=0;i--) rect(585+i*5,548-i*4,34,49,"#F5A565","#FFFFFF",1,1);
txt("Fkin ∈ ℝ B×C×H×W",620,629,13,"#172033",700);
arrow(638,465,638,395,"#F07A24");
multiline(["kinetic bias +","difference refinement"],690,407,12,"#F07A24",700,16);

rect(755,215,125,260,"#FFFFFF","#F07A24",14,2);
txt("DUAL-ROLE FUSION",817,208,15,"#F07A24",700);
rect(768,255,99,175,"#FFF1E6","#F07A24",12,2);
multiline(["Kinetic–Raw","Residual Fusion"],817,292,17,"#F07A24",700,21);
txt("Fphase + Fkin",817,351,13,"#172033",700);
parts.push(`<circle cx="817" cy="390" r="15" fill="#FFFFFF" stroke="#F07A24" stroke-width="2"/>`);
txt("+",817,397,22,"#F07A24",700);
txt("F0",817,424,14,"#172033",700);
parts.push(`<polyline points="714,550 742,550 742,390 768,390" fill="none" stroke="#F07A24" stroke-width="3" marker-end="url(#arrow-F07A24)"/>`);
txt("explicit kinetic feature",700,539,11,"#F07A24",700);
arrow(720,250,768,300,"#2463C7");

rect(900,95,350,580,"#FFFFFF","#7A4CC2",14,2);
txt("HYBRID MULTISCALE ENCODING",1075,88,17,"#7A4CC2",700);
txt("CNN Encoder",988,140,17,"#2463C7",700);
txt("U-Net Decoder",1160,140,17,"#168B91",700);
[["C · H",158,130],["2C · H/2",225,120],["4C · H/4",292,110],["8C · H/8",359,100],["16C · H/16",426,90]].forEach(([s,y,w],i)=>{rect(925,y,w,38,i===4?"#F2ECFF":"#EAF2FF",i===4?"#7A4CC2":"#2463C7",8,1.5);txt(s,925+w/2,y+25,14,"#172033",700);});
rect(915,510,150,70,"#F2ECFF","#7A4CC2",10,2);
txt("Swin Transformer Bottleneck",990,536,14,"#7A4CC2",700);
txt("depth=2 · window=7 · heads=4",990,559,12,"#172033",600);
[["8C",426],["4C",359],["2C",292],["C",225]].forEach(([s,y])=>{rect(1120,y,78,38,"#E8F7F6","#168B91",8,1.5);txt(s,1159,y+25,14,"#172033",700);});
[177,244,311,378].forEach((y,i)=>parts.push(`<path d="M ${1055-i*10} ${y} H 1110 V ${244+i*67}" fill="none" stroke="#AAB6C8" stroke-width="2" stroke-dasharray="5 4"/>`));
arrow(880,390,912,390,"#172033");

rect(1270,95,310,580,"#FFFFFF","#D94747",14,2);
txt("UNCERTAINTY-AWARE REFINEMENT",1425,88,17,"#D94747",700);
rect(1290,132,270,66,"#F5F7FA","#172033",10,1.5);
txt("Coarse Segmentation Head",1425,158,17,"#172033",700);
txt("Pc = σ(Conv1×1(Fdec))",1425,184,13,"#526078");
arrow(1425,199,1425,220,"#D94747",2);
rect(1290,224,270,72,"#FFF0F0","#D94747",10,2);
txt("Uncertainty Map",1425,251,18,"#D94747",700);
txt("U = 1 − |2Pc − 1|",1425,280,17,"#172033",700);
arrow(1425,297,1425,321,"#D94747",2);
rect(1290,325,270,105,"#FFF0F0","#D94747",10,2);
multiline(["Uncertainty-Guided","Boundary Refinement"],1425,356,19,"#D94747",700,23);
txt("decoder feature + uncertainty gating",1425,412,12,"#526078");
arrow(1199,244,1278,377,"#168B91");
[["Tumor","Segmentation","#2463C7","#EAF2FF"],["Boundary","Prediction","#D94747","#FFF0F0"],["Uncertainty","Supervision","#F07A24","#FFF1E6"]].forEach(([a,b,c,f],i)=>{const x=1290+i*92;arrow(x+43,431,x+43,460,c,2);rect(x,465,86,145,f,c,10,1.5);multiline([a,b],x+43,490,14,c,700,18);parts.push(`<ellipse cx="${x+43}" cy="555" rx="20" ry="28" fill="${i===0?c:"#FFFFFF"}" stroke="${c}" stroke-width="${i===0?1:3}"/>`);});

rect(225,725,1030,125,"#F5F7FA","#AAB6C8",12,1.5);
txt("Overall training objective",245,758,17,"#172033",700,"start");
txt("L = LDice+BCE + λb Lboundary + λu Luncertainty + λa Lattention-smooth",740,801,23,"#172033",700);
[["#2463C7","appearance feature"],["#F07A24","kinetic guidance"],["#7A4CC2","transformer"],["#168B91","decoder"],["#D94747","auxiliary output"]].forEach(([c,s],i)=>{rect(260+i*185,818,18,18,c,c,4,0);txt(s,283+i*185,833,12,"#526078",400,"start");});
rect(1270,725,310,125,"#FFFFFF","#AAB6C8",12,1.5);
txt("Key contribution",1285,758,17,"#172033",700,"start");
multiline(["Kinetic priors guide phase attention","and explicitly enter the encoder","through residual fusion."],1425,786,15,"#526078",400,20);
parts.push(`</svg>`);

fs.mkdirSync(path.dirname(out), { recursive: true });
fs.writeFileSync(out, parts.join("\n"), "utf8");
console.log(out);
