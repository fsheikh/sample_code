#!/bin/python
from flask import Flask
from flask import abort
from flask import jsonify
from subprocess import call

# Derived from http://blog.miguelgrinberg.com/post/designing-a-restful-api-with-python-and-flask
# NO COPYRIGHT infringement intended


# Server prototype maintains a tiny database as a list of dictionaries
# Need to evolve it beyond prototype
student_data = [{'id':301, 'name':'Shafay Sheikh', 'sclass':3, 'section': 'D'},
                {'id':310, 'name':'Marium Batool', 'sclass':3, 'section': 'E'},
                {'id':299, 'name':'Shahid Mir',    'sclass':2, 'section': 'F'},
                {'id':162, 'name':'Fareha Butt',   'sclass':1, 'section': 'B'},
                {'id':345, 'name':'Abeera Babar',  'sclass':3, 'section': 'C'}]

flite_voice_command="/home/fahim/flite -voice /home/fahim/voices/cmu_indic_aup_mr.flitevox -t "
outfilename="/home/fahim/out.wav"

for data in student_data:
    print data['id'], data['name'], data['sclass'], data['section']

app = Flask(__name__)

@app.route('/')
def index():
    return "Welcome to Secure School Announcer!"

# Server needs to queue the incoming student IDs from sender client
# then when the receiving client comes with a get request it knows
# top of queue needs to be returned. TODO Queue implementation

@app.route('/ssa/lgs-jt/<int:student_id>', methods=['GET'])
def get_student_data(student_id):
    sdata = [data for data in student_data if data['id'] == student_id]
    if len(sdata)==0:
        abort(404)
    t2a='"'+sdata[0]['name']+' '+str(sdata[0]['sclass'])+' '+sdata[0]['section']+'"'
    fcommand = flite_voice_command++" -o "+outfilename
    print t2a
    print fcommand
    call(fcommand, shell=True)
    call("/usr/bin/aplay -v "+outfilename, shell=True)
    return jsonify({'student': sdata[0]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
