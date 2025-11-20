import subprocess

def run(script):
    print(f"\n==============================")
    print(f"Running {script} ...")
    print(f"==============================\n")
    result = subprocess.run(["python", script])
    if result.returncode != 0:
        print(f"❌ Error running {script}, stopping execution.")
        exit(1)

# STEP 1 — ONLY if you need to rebuild dataset splits
run("pre_process.py")

# STEP 2 — Training
run("main_train.py")

# STEP 3 — Evaluation
run("main_eval.py")

print("\n🎉 ALL DONE SUCCESSFULLY!\n")
