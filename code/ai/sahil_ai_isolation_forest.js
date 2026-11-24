// صهيل — نموذج Isolation Forest (نفس الكود المضمّن في واجهة الجهاز)
// المدخلات: أيام الخيل [التاريخ، اليوم، ...، الأكل% (7)، ...، الشرب% (10)، ...، الحرارة (12)، النبض (13)، الحالة (14)]
// المخرجات لكل يوم: s = درجة الشذوذ (0-1)، lvl = 0 طبيعي / 1 تنبيه مبكر / 2 تنبيه، why = أسباب التنبيه

const AI=(()=>{
 const rng=s=>()=>{s|=0;s=s+0x6D2B79F5|0;let t=Math.imul(s^s>>>15,1|s);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296};
 const cF=n=>n>2?2*(Math.log(n-1)+0.5772156649)-2*(n-1)/n:n==2?1:0;
 function forest(X,trees=200,seed=11){
  const r=rng(seed),n=X.length,ss=Math.min(64,n),lim=Math.ceil(Math.log2(ss)),dim=X[0].length,T=[];
  const build=(rows,dp)=>{if(dp>=lim||rows.length<=1)return{s:rows.length};const f=Math.floor(r()*dim);let mn=1e9,mx=-1e9;for(const x of rows){if(x[f]<mn)mn=x[f];if(x[f]>mx)mx=x[f]}
   if(mx-mn<1e-9)return{s:rows.length};const v=mn+r()*(mx-mn),L=[],R=[];for(const x of rows)(x[f]<v?L:R).push(x);return{f,v,l:build(L,dp+1),r:build(R,dp+1)}};
  for(let t=0;t<trees;t++){const s=[];for(let i=0;i<ss;i++)s.push(X[Math.floor(r()*n)]);T.push(build(s,0))}
  const path=(nd,x,d)=>nd.f===undefined?d+cF(nd.s):path(x[nd.f]<nd.v?nd.l:nd.r,x,d+1),c=cF(ss);
  return x=>Math.pow(2,-T.reduce((a,t)=>a+path(t,x,0),0)/T.length/c)}
 const med=a=>{const s=[...a].sort((x,y)=>x-y),m=s.length>>1;return s.length%2?s[m]:(s[m-1]+s[m])/2};
 // features: food%, water%, body temp, pulse, water drop vs previous 3 days
 const NAMES=['الأكل','الشرب','حرارة الخيل','النبض','تراجع الشرب'];
 const DIR=[-1,-1,1,1,-1];          // direction that means concern
 function feats(days){return days.map((d,i)=>{const p=days.slice(Math.max(0,i-3),i);const pw=p.length?p.reduce((a,x)=>a+x[10],0)/p.length:d[10];return[d[7],d[10],d[12],d[13],d[10]-pw]})}
 function analyze(days,extra){
  const ex=extra?(Array.isArray(extra[0])?extra:[extra]):[],all=days.concat(ex),F=feats(all),train=F.slice(0,days.length);
  if(train.length<7)return null;
  const score=forest(train),cols=train[0].map((_,j)=>train.map(r=>r[j]));
  const M=cols.map(med),S=cols.map((c,j)=>Math.max(1.4826*med(c.map(v=>Math.abs(v-M[j]))),[3,3,0.15,1.5,6][j]));
  return F.map((x,i)=>{const s=score(x),z=x.map((v,j)=>(v-M[j])/S[j]*DIR[j]);
   const why=z.map((v,j)=>[v,j]).filter(([v])=>v>=2).sort((a,b)=>b[0]-a[0]);
   const concern=why.length>0;
   const lvl=concern&&s>=0.68?2:concern&&s>=0.56?1:0;
   return{s,lvl,why:why.slice(0,2).map(([v,j])=>({j,name:NAMES[j],pct:j<2||j==4?Math.round(x[j]-M[j]):null,val:x[j],base:M[j]}))}})}
 return{analyze}})();
if(typeof module!=='undefined')module.exports=AI;
