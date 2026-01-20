"""MAIN PIPELINE RUNNER

This script runs the entire web mining pipeline in sequence.
Use this for convenient end-to-end execution.

Usage:
    python run_pipeline.py
"""

import subprocess
import sys
import os

# Get project directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = script_dir

def run_script(script_name, step_number, step_name):
    """Run a script and report results."""
    print("\n" + "="*70)
    print(f"Running: {step_name}")
    print("="*70 + "\n")
    
    script_path = os.path.join(project_dir, 'scripts', script_name)
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=project_dir,
            check=True,
            capture_output=False
        )
        print(f"\n✓ Step {step_number} completed successfully!\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Step {step_number} failed with error code {e.returncode}\n")
        return False
    except Exception as e:
        print(f"\n✗ Error running step {step_number}: {e}\n")
        return False

def main():
    """Run complete pipeline."""
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  WEB MINING PROJECT - COMPLETE PIPELINE".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70 + "\n")
    
    steps = [
        ('preprocess.py', 1, 'TEXT PREPROCESSING'),
        ('topic_model.py', 2, 'TOPIC MODELING (LDA)'),
        ('cluster.py', 3, 'K-MEANS CLUSTERING'),
        ('visualize.py', 4, 'VISUALIZATION'),
    ]
    
    completed = 0
    failed = 0
    
    for script, step_num, step_name in steps:
        if run_script(script, step_num, step_name):
            completed += 1
        else:
            failed += 1
            print(f"Pipeline stopped at step {step_num}")
            break
    
    # Final summary
    print("\n" + "="*70)
    print("PIPELINE EXECUTION SUMMARY")
    print("="*70)
    print(f"Completed steps: {completed}/{len(steps)}")
    print(f"Failed steps: {failed}")
    
    if failed == 0 and completed == len(steps):
        print("\n✓ ALL STEPS COMPLETED SUCCESSFULLY!")
        print("\nOutput files saved to:")
        print(f"  - Data: {os.path.join(project_dir, 'data')}/")
        print(f"  - Results: {os.path.join(project_dir, 'results')}/")
        print("\nProject is ready for analysis and report writing.\n")
        return 0
    else:
        print(f"\n✗ Pipeline incomplete. Please check the error messages above.\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
