import subprocess
import re

from gavel.logic.status import Status, get_status

LEO_PATH = "./leo3"

def prove(tptp_problem: str, timeout=60) -> Status:
    """Prove the given TPTP problem using the LEO III theorem prover (https://github.com/leoprover/Leo-III/tree/v1.7.18).
    
    Args:
        tptp_problem: A string containing the TPTP problem to prove.
        
    Returns:
        A Status object representing the result of the proof attempt.
        
    The function passes the TPTP problem text directly to the LEO prover through stdin using the '-' parameter.
    It then processes the output to extract the SZS status and returns an appropriate Status object.
    If no status is found in the output, it returns a Status with "Unknown".
    """
    # Run the LEO prover with the '-' parameter to read from stdin
    import os
    print(f"Current directory: {os.getcwd()}")
    raw_result = subprocess.run(
        [LEO_PATH, "-", "-t", str(timeout)],
        input=tptp_problem,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    
    # Process the output to determine the result
    output = raw_result.stdout
    
    # Look for SZS status in the output
    status_match = re.search(r'% SZS status (\w+)', output)
    if status_match:
        status_str = status_match.group(1)
        if "Error" in status_str:
            print(f"Error in proof attempt: {output}")
        # Return appropriate Status object based on the result
        return get_status(status_str)
    else:
        # If no status found, return an error or unknown status
        print("No SZS status found in the output.")
        print(f"Output was: {output}")
        return get_status("Unknown")