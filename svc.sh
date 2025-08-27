cd /etc/systemd/system

gedit frpc.service  

[Unit]
Description=frpc service
After=network.target
 
[Service]
Type=simple
ExecStart=/root/frp/frpc -c /root/frp/frpc.ini
 
[Install]
WantedBy=multi-user.target

sudo systemctl enable frpc
sudo systemctl start frpc