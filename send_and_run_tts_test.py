import paramiko

def fetch_logs():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    print("Connecting to 173.214.172.254...")
    try:
        client.connect("173.214.172.254", username="root", password="4uF&V@FJ", allow_agent=False, look_for_keys=False)
        
        print("Executing cURL to test scheduling endpoint...")
        
        cmd = """curl -X POST 'http://173.214.172.254:8000/api/chat/query-with-voice' -H "Content-Type: application/json" -d '{"query": "مواعيد دكتور ميلاد عبده", "max_results": 5}'"""
        
        stdin, stdout, stderr = client.exec_command(cmd)
        out = stdout.read().decode('utf-8')
        err = stderr.read().decode('utf-8')
        
        print(out[:500] + "\n...[truncated binary data]" if "audio_data" in out else out)
        if err and "curl: " in err:
            print(f"STDERR: {err}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        client.close()

if __name__ == "__main__":
    fetch_logs()
