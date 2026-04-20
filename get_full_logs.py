import paramiko
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect("173.214.172.254", username="root", password="4uF&V@FJ", allow_agent=False, look_for_keys=False)
stdin, stdout, stderr = client.exec_command("docker logs heal-query-hub-backend-1 --tail 300 2>&1")
logs = stdout.read().decode('utf-8')
with open('backend_logs.txt', 'w') as f:
    f.write(logs)
client.close()
