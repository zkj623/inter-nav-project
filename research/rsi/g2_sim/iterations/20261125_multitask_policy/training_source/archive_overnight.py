from pathlib import Path
import shutil,json,hashlib,subprocess,datetime
W=Path(__file__).resolve().parent;R=W.parent/'g2-act-threecam-20261124';P=W.parent/'g2-publish-20261118';O=P/'research/rsi/g2_sim/iterations/20261125_multitask_policy';O.mkdir(parents=True,exist_ok=True)
files={}
for kind,paths in [('training_source',list(W.glob('*.py'))),('simulation_source',[p for p in R.glob('*.py') if p.name.startswith(('policy_','threecam_recorder','plan_wide','run_wide','wide_posture','verify_threecam'))])]:
 d=O/kind;d.mkdir(exist_ok=True)
 for p in paths:
  shutil.copy2(p,d/p.name);files[str((d/p.name).relative_to(O))]=hashlib.sha256(p.read_bytes()).hexdigest()
for repo in ['act','openpi']:
 d=W/'vendor'/repo
 (O/(repo+'_upstream.json')).write_text(json.dumps(dict(remote=subprocess.check_output(['git','remote','get-url','origin'],cwd=d,text=True).strip(),commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=d,text=True).strip()),indent=2))
 (O/(repo+'_local.patch')).write_bytes(subprocess.check_output(['git','diff','HEAD'],cwd=d))
dest=O/'results';dest.mkdir(exist_ok=True)
for p in W.glob('*.json'):
 if p.stat().st_size<2_000_000:shutil.copy2(p,dest/p.name)
for folder in ['datasets','evaluations','runs','pi05_sources']:
 for p in (W/folder).rglob('*'):
  if p.is_file() and p.suffix in ['.json','.jsonl','.py'] and p.stat().st_size<10_000_000:
   out=dest/folder/p.relative_to(W/folder);out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,out)
(O/'source_manifest.json').write_text(json.dumps(dict(snapshot_time=datetime.datetime.now().isoformat(),sha256=files,remote_workspace=str(W),simulation_root=str(R)),indent=2))
(O/'.gitattributes').write_text('training_source/*.py -text whitespace=cr-at-eol,-blank-at-eof\nsimulation_source/*.py -text whitespace=cr-at-eol,-blank-at-eof\nresults/**/*.py -text whitespace=cr-at-eol,-blank-at-eof\n')
(O/'README.md').write_text('''# G2 澶氱墿浣?ACT / 蟺0.5 璁粌瀹為獙

鏈洰褰曚繚瀛樿缁冦€佹暟鎹噰闆嗐€侀棴鐜墽琛屽拰鐙珛楠屾敹浠ｇ爜銆傜姸鎬佷互 results/evaluations/*/summary.json 涓哄噯锛沴oss 闄嶄綆鎴栦笓瀹跺洖鏀鹃€氳繃閮戒笉绛変簬绛栫暐鎴愬姛銆?
## 浠诲姟涓庢帴鍙?
鍚屼竴妯″瀷鎺у埗澶氫釜鐗╀綋鐨勫弻鑷傛姄鍙栨姮鍗囥€傚簳鐩樸€佽叞鑵垮拰澶撮儴鍥哄畾鍦ㄥ叡鍚屼綆浣嶅Э鎬侊紝杈撳叆澶撮儴涓庡乏鍙宠厱閮?RGB锛堢瓑姣斾緥缂╂斁骞惰ˉ杈硅嚦 224脳224锛夈€?6 缁村疄闄呭叧鑺傜姸鎬侊紝浠ュ強 ACT 鐨勭墿浣撲换鍔＄紪鍙锋垨 蟺0.5 鐨勬枃瀛楁彁绀恒€傝緭鍑?50 姝ュ姩浣滃潡锛屾瘡娆℃墽琛屽墠 10 姝ワ紝鍐嶈娴嬨€傛帶鍒堕鐜?20 Hz锛岀墿鐞嗕豢鐪?120 Hz銆備袱鑷傚叡14鍏宠妭鍔?涓す鐖€?
鎵ц鍣ㄦ帴鏀剁粷瀵瑰叧鑺傜洰鏍囷紱鐩稿鍔ㄤ綔瀹為獙鍦ㄦā鍨嬪唴閮ㄩ娴嬬浉瀵瑰綋鍓嶅叧鑺傜姸鎬佺殑14缁村弻鑷傜洰鏍囷紝鎵ц鍓嶅姞鍥炲綋鍓嶇姸鎬侊紝澶圭埅淇濇寔缁濆鐩爣銆傛墍鏈夊叧鑺傞『搴忋€佺粺璁″€煎拰鍔ㄤ綔琛ㄧず淇濆瓨鍦ㄥ悇 checkpoint 涓€傝鍕挎贩鐢ㄧ粺璁″€笺€佺浉鏈洪『搴忓拰鍔ㄤ綔琛ㄧず銆?
## 鏁版嵁

鍘熷鐩爣涓?2绉嶇墿浣撳悇20鏉￠€氳繃楠屾敹鐨勬紨绀恒€傛墿澶ц寖鍥撮噰闆嗛澶栧皾璇?XY 鍚劼?cm銆侀儴鍒喡?掳鏃嬭浆锛涘悗缁噰鏍疯寖鍥磋 plan_wide_threecam.py銆傚彧鎶婄嫭绔嬮獙鏀堕€氳繃鐨勮建杩圭撼鍏ヨ缁冦€俻ilot4 鏄?绉嶇墿浣撳悇2鏉＄殑杩囨嫙鍚堣瘖鏂泦锛屾病鏈夋祴璇曢泦娉涘寲缁撹銆俥xpanded12 鏄彟涓€涓喕缁撳揩鐓э紝鍏蜂綋鏍锋湰鍙婂垝鍒嗚 manifest.json锛屾湭杈惧埌240鏉℃椂涓嶅彲绉板畬鏁存暟鎹泦銆?
## 鎵ц涓庨獙璇?
浣跨敤鍚屼竴 checkpoint SHA 鍦ㄤ笉鍚岀墿浣撲笂杩愯妯″瀷闂幆锛涗笉璇诲彇涓撳鍔ㄤ綔銆佺墿浣撲綅濮挎垨涓撳鎺ヨЕ鍙嶉鏉ユ帶鍒躲€傚悇鐗╀綋娌跨敤宸叉湁鐨勪綆灞傞┍鍔ㄩ厤缃紝灏氭湭楠岃瘉缁熶竴椹卞姩鍙傛暟銆傝瘎浼扮户鎵垮師濮嬬鎾炪€佸叧鑺傘€佸彲瑙嗚〃闈㈡帴瑙︺€佹姮鍗囬珮搴︺€佺ǔ瀹氫繚鎸佺瓑妫€鏌ャ€俤iagnostic_expert_command_replay 鏄崟鐙殑鎵ц鎺ュ彛璇婃柇锛岀粷涓嶈鍏ュ涔犳垚鍔熺巼銆?
姝ゅ鏄豢鐪熷疄楠岋紝鍥哄畾韬綋鐨勯珮鎵煩璁惧畾涓嶄唬琛ㄧ湡鏈鸿叞鑵胯兘鍔涖€傚師濮?HDF5銆佽棰戝強鏉冮噸淇濈暀鍦ㄦ湇鍔″櫒锛屼笉鎻愪氦鍒?Git銆?
## 鏈嶅姟鍣ㄨ矾寰?
- 璁粌宸ヤ綔鍖猴細/data20t/embodied/zkj/inter-nav-results/g2-policy-overnight-20261124
- 浠跨湡涓庡師濮嬮噰闆嗭細/data20t/embodied/zkj/inter-nav-results/g2-act-threecam-20261124
- ACT锛歳uns/<run>/step_<step>.pt
- 蟺0.5锛歝heckpoints/g2_pi05_lora/<run>/<step>
- 瑙嗛锛歴ummary.json 涓悇鏉＄洰鐨?video 瀛楁

璁粌涓庤瘎浼板惎鍔ㄥ弬鏁拌 training_source 涓殑 curriculum/watch 鑴氭湰銆備笉瑕佺洿鎺ラ噸鏂拌繍琛屾暣涓紪鎺掕剼鏈鐩栫幇鏈夊疄楠岋紱鏂板疄楠岃浣跨敤鏂扮殑 run 鍚嶇О銆?''',encoding='utf-8')
print(O)

