let data, si=0, ti=0, mapImage=new Image(), regionImage=new Image();
const el=id=>document.getElementById(id), canvas=el('map'), ctx=canvas.getContext('2d');
const xy=p=>p.map(v=>v.toFixed(2)).join(', '), scene=()=>data.scenes[si], task=()=>scene().tasks[ti];
const bands={short:'短距离',medium:'中距离',long:'长距离'};
const colors=['#2166ac','#b24a71','#218776','#8c62a6','#bf702a','#537a31','#b63b34','#557d95','#9e8023','#3f6684'];
const toolbar=document.querySelector('.toolbar');
toolbar.insertAdjacentHTML('afterbegin','<label><input id="allTasks" type="checkbox" checked>全部任务总览</label>');
toolbar.insertAdjacentHTML('beforeend','<label><input id="endpointRegions" type="checkbox">端点区域</label>');
const zoom=document.createElement('button');zoom.textContent='放大查看';
zoom.onclick=()=>{zoom.textContent=canvas.classList.toggle('zoomed')?'适应窗口':'放大查看'};toolbar.appendChild(zoom);
const distribution=document.createElement('div');distribution.id='distribution';distribution.className='notice';
document.querySelector('.figure').insertAdjacentElement('afterend',distribution);
el('objects').closest('table').querySelector('thead tr').insertAdjacentHTML('beforeend','<th>所选清障方案使用次数</th>');

function loadScene(i){
 si=i;ti=0;el('allTasks').checked=true;
 document.querySelectorAll('#scenes button').forEach((b,j)=>b.classList.toggle('active',i===j));
 const s=scene();el('sceneTitle').textContent=`${s.id} · ${s.floor_area_m2.toFixed(1)} m² · ${s.objects.length} 个固定物体`;
 mapImage.src=`${s.id}/map.png`;regionImage.src=`${s.id}/connectivity.png`;
 canvas.width=s.map.width*2;canvas.height=s.map.height*2;
 el('objects').innerHTML=s.objects.map((o,j)=>`<tr><td>${o.instance_id} ${o.name}</td><td>${o.asset_id}</td><td>${xy(o.position_m.slice(0,2))}</td><td>${o.yaw_deg}°</td><td>${xy(o.parking_position_m.slice(0,2))}</td><td>${s.tasks.filter(t=>(t.optimistic_relocation_subset||[]).includes(j)).length}</td></tr>`).join('');
 el('source').textContent=`GRScenes 场景：${s.scene_id} · 物体来源：genie2-rsi @ c9ce1a5 · 初始静置测试：${s.physics_validation==='passed_initial_passive_settling'?'已通过':s.physics_validation==='failed_initial_passive_settling'?'未通过':'待验证'}`;
 const v=s.task_diversity,b=v.length_band_counts;
 const topo=s.layout_topology;
 distribution.innerHTML=`<b>任务分布</b>：短 ${b.short||0} / 中 ${b.medium||0} / 长 ${b.long||0}；${v.unique_endpoint_positions} 个不同端点位置。<br>${topo?`物体阻塞后，任务涉及 ${topo.task_component_count} 个初始连通区域；交互任务覆盖 ${Object.keys(topo.required_task_region_pairs).length} 组区域连接。<br>`:''}C 表示物体未移动时的连通区域；R 表示用于统计端点分布的空间分区。<a href="layout_validation.json">阻塞布局验证</a> · <a href="task_diversity.json">任务分布</a>`;
 renderTasks();draw();
}
function selectTask(i){ti=i;el('allTasks').checked=false;renderTasks();draw()}
function renderTasks(){
 const s=scene(),t=task(),all=el('allTasks').checked;
 el('tasks').innerHTML=s.tasks.map((x,j)=>`<button class="${!all&&j===ti?'active':''}" onclick="selectTask(${j})"><span><i class="swatch" style="--c:${colors[j]}"></i>${x.id}<small>${bands[x.length_band]} · ${x.class==='interaction_required'?'必须交互':'无需交互'}</small><small>${x.initial_start_region||x.start_region} → ${x.initial_goal_region||x.goal_region}</small></span><span>${x.cleared_route_length_m.toFixed(1)} m</span></button>`).join('');
 el('detail').innerHTML=all?'总览同时展示 10 个任务。<br>同色线和编号对应同一任务。<br>● 起点（S）　◆ 终点（G）<br>勾选初始连通区域，可检查物体封堵效果。<br>点击任务可查看所选清障物体。':`<strong>${t.cleared_route_length_m.toFixed(2)} m</strong> 参考路线<br>${bands[t.length_band]} · ${t.initial_start_region||t.start_region} → ${t.initial_goal_region||t.goal_region}<br>起点：(${xy(t.start_position_m.slice(0,2))})<br>终点：(${xy(t.goal_position_m.slice(0,2))})<br>候选终态下最少搬移：${t.minimum_relocations_in_optimistic_model} 件<br>所选清障物体：${(t.optimistic_relocation_subset||[]).map(j=>s.objects[j].instance_id).join('、')||'无'}<br><a href="${s.id}/episodes.json">该场景配置</a>`;
}
function draw(){
 if(!data||!mapImage.complete||!mapImage.naturalWidth)return;
 const s=scene(),all=el('allTasks').checked,parked=el('state').value==='cleared',scale=2/s.map.resolution;
 const px=p=>[(p[0]-s.map.origin[0])*scale,(p[1]-s.map.origin[1])*scale];
 ctx.clearRect(0,0,canvas.width,canvas.height);ctx.imageSmoothingEnabled=false;
 const im=el('regions').checked&&regionImage.complete&&regionImage.naturalWidth?regionImage:mapImage;
 ctx.drawImage(im,0,0,canvas.width,canvas.height);
 function label(text,p,color,size=16){ctx.font=`bold ${size}px sans-serif`;ctx.textAlign='center';ctx.lineWidth=4;ctx.strokeStyle='white';ctx.strokeText(text,p[0],p[1]);ctx.fillStyle=color;ctx.fillText(text,p[0],p[1])}
 function object(o,parking,ghost){
  const p=px(parking?o.parking_position_m:o.position_m),w=o.dimensions_m[0]*scale,h=o.dimensions_m[1]*scale;
  ctx.save();ctx.translate(...p);ctx.rotate(o.yaw_deg*Math.PI/180);ctx.fillStyle=parking?'#428678':'#ce803c';ctx.strokeStyle=ctx.fillStyle;ctx.lineWidth=2;ctx.setLineDash(ghost?[6,5]:[]);
  if(!ghost){ctx.globalAlpha=.8;ctx.fillRect(-w/2,-h/2,w,h)}ctx.globalAlpha=1;ctx.strokeRect(-w/2,-h/2,w,h);ctx.restore();
  if(!ghost)label(o.instance_id,[p[0],p[1]-17],'#172e33');
 }
 s.objects.forEach(o=>{object(o,!parked,true);object(o,parked,false)});
 const shown=all?s.tasks.map((t,i)=>[t,i]):[[task(),ti]];
 if(el('route').checked)for(const [t,i] of shown){
  ctx.beginPath();ctx.strokeStyle=colors[i];ctx.lineWidth=all?3:4;ctx.globalAlpha=all?.72:1;ctx.setLineDash(!parked&&t.class==='interaction_required'?[9,6]:[]);
  t.reference_route_xy_m.forEach((p,j)=>{const q=px(p);j?ctx.lineTo(...q):ctx.moveTo(...q)});ctx.stroke();ctx.setLineDash([]);ctx.globalAlpha=1;
 }
 for(const [t,i] of shown){
  const start=px(t.start_position_m),goal=px(t.goal_position_m);ctx.fillStyle=colors[i];ctx.strokeStyle='white';ctx.lineWidth=2;
  ctx.beginPath();ctx.arc(...start,all?7:9,0,Math.PI*2);ctx.fill();ctx.stroke();
  ctx.beginPath();ctx.moveTo(goal[0],goal[1]-9);ctx.lineTo(goal[0]+9,goal[1]);ctx.lineTo(goal[0],goal[1]+9);ctx.lineTo(goal[0]-9,goal[1]);ctx.closePath();ctx.fill();ctx.stroke();
  label(`${i+1}S`,[start[0],start[1]-12],colors[i]);label(`${i+1}G`,[goal[0],goal[1]+22],colors[i]);
 }
 if(el('endpointRegions').checked)for(const region of s.endpoint_regions){const p=px(region.center_xy_m);label(region.id,p,'#303537',22)}
 if(el('regions').checked)for(const region of s.component_regions||[]){label(region.name,px(region.center_xy_m),'#1c3037',22)}
 ctx.strokeStyle='#213237';ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(20,canvas.height-25);ctx.lineTo(20+scale,canvas.height-25);ctx.stroke();ctx.fillStyle='#213237';ctx.textAlign='left';ctx.font='15px sans-serif';ctx.fillText('1 m',20,canvas.height-34);
 el('stateHint').textContent=all?'任务总览：展示端点分布和参考路线。必须交互任务采用清障后的参考路线，对照任务采用初始状态的路线；这些线并不表示当前显示状态下均可通行。':parked?'清障参考状态：物体处于候选移开位置。这是几何终态，不是机器人执行结果。':task().class==='interaction_required'?'初始固定布局：起终点不连通。虚线是清障后的参考路线，当前状态不能直接沿它通行。':'初始固定布局：实线是无需移动物体即可通行的最短参考路线。';
}
mapImage.onload=draw;regionImage.onload=draw;
['state','regions','route','endpointRegions'].forEach(id=>el(id).addEventListener('change',draw));
el('allTasks').addEventListener('change',()=>{renderTasks();draw()});
fetch('viewer_data.json').then(r=>r.json()).then(d=>{data=d;el('scenes').innerHTML=d.scenes.map((s,i)=>`<button onclick="loadScene(${i})">${s.id} · ${s.floor_area_m2.toFixed(1)} m²<small>${{small:'小型',medium:'中型',large:'大型'}[s.size_band]}住宅 · 10 个任务</small></button>`).join('');loadScene(0)}).catch(e=>{el('scenes').innerHTML='<span class="error">加载失败，请刷新页面。</span>';console.error(e)});
fetch('geometry_validation.json').then(r=>r.json()).then(v=>el('validation').textContent=`几何验证：${v.passed?'通过':'未通过'}；${v.multi_relocation_tasks} 个任务在候选终态组合中至少搬移 2 件物体。必须交互任务参考路线：${v.required_path_range_m.map(x=>x.toFixed(2)).join('–')} m。`);
fetch('coverage.json').then(r=>r.json()).then(c=>{const section=document.createElement('section');section.className='panel below';section.innerHTML='<details><summary>15 类物体的场景与任务覆盖</summary><p class="hint">统计所选最少搬移方案使用各物体的任务数。物体在场景中出现，不等于每个任务都必须与它交互。</p><table><thead><tr><th>物体类型</th><th>所在场景</th><th>所选清障方案使用任务数</th></tr></thead><tbody>'+Object.entries(c).map(([id,v])=>`<tr><td>${v.name}<br><small>${id}</small></td><td>${v.scenes.join('、')}</td><td>${v.tasks_using_object_in_one_recorded_relocation_witness.length}</td></tr>`).join('')+'</tbody></table></details>';document.querySelector('main').insertBefore(section,document.querySelector('footer'))});
