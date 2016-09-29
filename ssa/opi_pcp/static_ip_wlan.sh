#!/bin/bash


# Assumes hostapd and dnsmasq have been properly configured and
# correctly running to enable an Access point on target.
echo "Starting static IP configuration on WLAN0"
sudo service hostapd stop
sudo ifconfig wlan0 up
sudo ifconfig wlan0 10.0.0.1/24
sudo service hostapd start
echo "Done AP setup!"


