import paramiko

def fetch_logs():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    print("Connecting to 173.214.172.254...")
    try:
        client.connect("173.214.172.254", username="root", password="4uF&V@FJ", allow_agent=False, look_for_keys=False)
        print("Connected. Fetching logs...")
        stdin, stdout, stderr = client.exec_command("docker logs heal-query-hub-backend-1 --tail 5000")
        
        output = stdout.read().decode('utf-8')
        error = stderr.read().decode('utf-8')
        
        combined = output + "\n" + error
        lines = combined.split('\n')
        
        found = False
        for i, line in enumerate(lines):
            if "مواعيد دكتور ميلاد عبده" in line or "Traceback (most recent call last)" in line:
                start = max(0, i - 10)
                end = min(len(lines), i + 30)
                print(f"\n--- Found MATCH around line {i} ---")
                print("\n".join(lines[start:end]))
                found = True
        
        if not found:
            print("No mentions of the query or Traceback found in the last 5000 lines.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        client.close()

if __name__ == "__main__":
    fetch_logs()
