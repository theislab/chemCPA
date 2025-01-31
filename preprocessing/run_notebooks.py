import os
from pathlib import Path
import logging
import subprocess
from tqdm import tqdm
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_python_script(script_path, env_vars=None):
    """Execute a single Python script with optional environment variables."""
    try:
        logger.info(f"Running script: {script_path}")
        
        # Create a copy of the current environment
        env = os.environ.copy()
        # Update with any additional environment variables
        if env_vars:
            env.update(env_vars)
        
        # Execute the Python script using subprocess with real-time output
        process = subprocess.Popen(
            ['python', str(script_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=env  # Pass the modified environment
        )
        
        # Print output in real-time
        while True:
            output = process.stdout.readline()
            error = process.stderr.readline()
            
            if output:
                print(output.strip())
            if error:
                print(error.strip(), file=sys.stderr)
                
            # Check if process has finished
            if output == '' and error == '' and process.poll() is not None:
                break
                
        return_code = process.poll()
        
        if return_code == 0:
            logger.info(f"Successfully executed: {script_path}")
            return True
        else:
            logger.error(f"Error executing {script_path} (return code: {return_code})")
            return False
            
    except Exception as e:
        logger.error(f"Error executing {script_path}: {str(e)}")
        return False

def main():
    # Get preprocessing directory
    preprocessing_dir = Path("preprocessing")
    if not preprocessing_dir.exists():
        preprocessing_dir = Path(__file__).parent
    
    # Get all python files that start with a digit
    python_files = sorted([f for f in preprocessing_dir.glob("[0-9]*.py")])
    
    logger.info(f"Found {len(python_files)} Python scripts to execute")
    
    # Execute scripts in order
    results = []
    for script_path in tqdm(python_files, desc="Executing scripts"):
        # Special handling for the SMILES script
        if "4_sciplex_SMILES" in script_path.name:
            # Run with LINCS_GENES=True
            success_true = run_python_script(script_path, {"LINCS_GENES": "true"})
            results.append((f"{script_path.name} (LINCS_GENES=True)", success_true))
            
            # Run with LINCS_GENES=False
            success_false = run_python_script(script_path, {"LINCS_GENES": "false"})
            results.append((f"{script_path.name} (LINCS_GENES=False)", success_false))
        else:
            # Run other scripts normally
            success = run_python_script(script_path)
            results.append((script_path.name, success))
    
    # Print summary
    print("\nExecution Summary:")
    print("-----------------")
    for name, success in results:
        status = "✓ Success" if success else "✗ Failed"
        print(f"{name}: {status}")

if __name__ == "__main__":
    main()
