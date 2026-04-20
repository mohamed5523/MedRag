import paramiko

def fetch_logs():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    print("Connecting to 173.214.172.254...")
    try:
        client.connect("173.214.172.254", username="root", password="4uF&V@FJ", allow_agent=False, look_for_keys=False)
        print("Connected. Fetching logs...")
        
        # Capture stdout and stderr
        stdin, stdout, stderr = client.exec_command("docker logs heal-query-hub-backend-1 --tail 300")
        
        logs = stdout.read().decode('utf-8')
        err = stderr.read().decode('utf-8')
        
        with open('backend_logs.txt', 'w') as f:
            f.write(logs)
            f.write(err)
            
        print("Logs saved to backend_logs.txt")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        client.close()

if __name__ == "__main__":
    fetch_logs()
