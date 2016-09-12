#!/bin/python
import json
import requests
import argparse
from subprocess import call

flite_voice_command = " -voice voices/cmu_indic_aup_mr.flitevox"
parser = argparse.ArgumentParser()
parser.add_argument("studentID", help="Required: Sends student ID to server", type=int)
#
def announce(announcement_text):
    return "Welcome to Secure School Announcer!"
# 
def id_to_text(student_id):
	return jsonify({'student': sdata[0]})

if __name__ == '__main__':

    # Parse command line, get student ID
	args = parser.parse_args()
	print args.studentID

	# Send a get request to server
	r = requests.get("http://localhost:5000/ssa/lgs-jt/"+str(args.studentID))

	# Parse JSON response to get student name and section
	json_dict    = r.json()
	json_student = json_dict['student']
	json_section = json_student['section']
	json_class   = json_student['sclass']
	json_name    = json_student['name']
	flite_data   = json_name+str(json_class)+json_section
	print flite_data
	# Call flite to announce with appropriate string
	call("./flite.exe"+flite_voice_command+" -t '"+flite_data+"' play", shell=True)