from pathlib import Path
import ast
R=Path(__file__).resolve().parent;s=(R/'torquelow_single_firm_trial.py').read_text().replace("ROOT/'torquelow_firm_trials'","ROOT/'torquelow_budget_trials'")
old="""        robot._articulation_view.set_max_efforts((effort_limits-np.abs(compensation))[None,:])
        robot._articulation_view.set_joint_efforts(compensation[None,:])"""
new="""        # Waist: combine gravity feedforward with the implicit position PD
        # before the native +/-50Nm clamp. Biasing its position reference by
        # g/kp is algebraically equivalent to adding g to that SAME drive.
        # This preserves the full negative correction range as well.
        drive_references=vals.copy();manual_compensation=compensation.copy()
        feedback_limits=effort_limits-np.abs(compensation)
        for i,n in enumerate(names):
            if 'body_joint' in n:
                manual_compensation[i]=0.
                feedback_limits[i]=effort_limits[i]
                drive_references[i]+=compensation[i]/max(1.,kp[i])
        robot._articulation_view.set_max_efforts(feedback_limits[None,:])
        robot._articulation_view.set_joint_efforts(manual_compensation[None,:])"""
assert old in s;s=s.replace(old,new).replace('joint_positions=vals[active]','joint_positions=drive_references[active]');s=s.replace("            rec['gravity_torques_nm']=", "            rec['waist_drive_references_rad']={n:float(drive_references[i]) for i,n in enumerate(names) if 'body_joint' in n}\n            rec['waist_manual_feedforward_nm']={n:float(manual_compensation[i]) for i,n in enumerate(names) if 'body_joint' in n}\n            rec['waist_native_drive_caps_nm']={n:float(feedback_limits[i]) for i,n in enumerate(names) if 'body_joint' in n}\n            rec['gravity_torques_nm']=")
s=s.replace("    result['required_positive_contacts']=2","    result['required_positive_contacts']=2\n    result['waist_controller']='implicit PD reference bias g/kp, native full +/-50Nm clamp, zero added manual waist effort; desired physical pose and original acceptance unchanged'")
ast.parse(s);(R/'torquelow_single_budget_trial.py').write_text(s);print('COMBINED_STRICT50_CONTROLLER_READY')
