import paramiko

def check_env():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    print("Connecting to 173.214.172.254...")
    try:
        client.connect("173.214.172.254", username="root", password="4uF&V@FJ", allow_agent=False, look_for_keys=False)
        print("Connected. Fetching env...")
        stdin, stdout, stderr = client.exec_command("cat /opt/heal-query-hub/backend/.env")
        
        output = stdout.read().decode('utf-8')
        lines = output.split('\n')
        
        for line in lines:
            if "OPENAI_API_KEY" in line:
                print("OPENAI_API_KEY exists on VPS and starts with:", line.split("=")[1][:5])
            if "OPENAI_TTS_MODEL" in line:
                print("TTS Model:", line)
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        client.close()

if __name__ == "__main__":
    check_env()
