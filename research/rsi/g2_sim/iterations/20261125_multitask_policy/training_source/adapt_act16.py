from pathlib import Path
import subprocess,json
W=Path(__file__).resolve().parent;P=W/'vendor/act';f=P/'detr/models/detr_vae.py'
s=f.read_text()
if 'self.task_embedding' not in s:
 s=s.replace('nn.Linear(14, hidden_dim)','nn.Linear(state_dim, hidden_dim)').replace('state_dim = 14 # TODO hardcode','state_dim = 16 # G2: 7+7 arms and 2 grippers')
 s=s.replace('self.action_head = nn.Linear(hidden_dim, state_dim)','self.task_embedding = nn.Embedding(12, hidden_dim)\n        self.action_head = nn.Linear(hidden_dim, state_dim)',1)
 s=s.replace('def forward(self, qpos, image, env_state, actions=None, is_pad=None):','def forward(self, qpos, image, env_state, actions=None, is_pad=None, task_ids=None):',1)
 s=s.replace('bs, _ = qpos.shape','bs, _ = qpos.shape\n        task = self.task_embedding(task_ids)',1)
 s=s.replace('qpos_embed = self.encoder_joint_proj(qpos)','qpos_embed = self.encoder_joint_proj(qpos) + task',1)
 s=s.replace('proprio_input = self.input_proj_robot_state(qpos)','proprio_input = self.input_proj_robot_state(qpos) + task',1)
 f.write_text(s)
(W/'act_upstream_revision.json').write_text(json.dumps(dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=P,text=True).strip(),changes='16D G2 state/actions; task embedding added to proprioception and CVAE state token; shared policy across all tasks'),indent=2))
print('ACT16_TASK_CONDITIONING_READY')
