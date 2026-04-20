import paramiko

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('173.214.172.254', username='root', password='4uF&V@FJ', timeout=10)

stdin, stdout, stderr = client.exec_command("docker logs heal-query-hub-backend-1 --tail 2000 2>&1")
out = stdout.read().decode('utf-8', errors='replace')
with open('backend_logs.txt', 'w') as f:
    f.write(out)

client.close()
