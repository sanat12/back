from flask import Flask, request, jsonify, make_response ,send_file
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from helper import bcrypt,db
from flask_marshmallow import Marshmallow
from flask_cors import CORS
from functools import wraps
from helper import create_app
from ml import PredictDisease
from flask_login import current_user
import jwt
import json
import datetime
import os
import uuid
import base64
import re
from cost import HealthExpenditure 

app=create_app()

def token_required(func):
    def wrapper(*args,**kwargs):

        token = request.headers['x-auth-token']

        if not token:
            return jsonify({'message':'Token is missing'})
        
        try:
            data = jwt.decode(token,app.config['SECRET_KEY'])
            request.data = data
        except:
            return jsonify({'message':'invalid token'})
        return func(*args,**kwargs)
    wrapper.__name__ = func.__name__
    return wrapper



from models import User,Post

#auth decorator to act as middleware

@app.route("/",methods=["GET"])
def getpost():
    return jsonify({'message':"something"})

@app.route("/register",methods=["POST"])    
def register():
    #if request.method=='GET':
        #return jsonify({'message':"Get request made"})
    req = request.json
    if(req.get('username') and req.get('email') and req.get('password')):
        # print(req['email'])
        
        user = User.query.filter_by(email= req['email']).first()

        if(user):
            return jsonify({'message':'2'})

        password = bcrypt.generate_password_hash(req['password']).decode('utf-8')
        user1 = User(username=req['username'],email=req['email'],password=password,)
        db.session.add(user1)
        db.session.commit()
        token = jwt.encode({'id':user1.id,'exp': datetime.datetime.utcnow() + datetime.timedelta(minutes=180)},app.config['SECRET_KEY'])         
        # print("token:"+token.decode('UTF-8'))
        if token:
            resp = {
                'token': token.decode('UTF-8'),
                'user' : {
                'username':user1.username,
                    'email': user1.email,
                    'id' : user1.id
                }
            } 
            return jsonify(resp)
        else:
            return jsonify({'message':'3'})
    else:
        return jsonify({'message': '1'})
    

@app.route("/login",methods=["POST"])
def login():
    req = request.json
    if(req.get('email') and req.get('password')):
        user = User.query.filter_by(email= req['email']).first()
        if(user):
            if(bcrypt.check_password_hash(user.password,req['password'])):
                #things to do after checking the email and password
                token = jwt.encode({'id':user.id,'exp': datetime.datetime.utcnow() + datetime.timedelta(minutes=180)},app.config['SECRET_KEY'])         
                # print("token:"+token.decode('UTF-8'))
                if token:
                    resp = {
                        'token': token.decode('UTF-8'),
                        'user' : {
                            'username':user.username,
                            'email': user.email,
                            'id' : user.id
                        }
                    } 
                    return jsonify(resp)
                else:
                    return jsonify({'message':'3'})
            else:
                return jsonify({'message':'4'})
        else:
             return jsonify({'message':'2'})
    else:
        return jsonify({'message':'1'})

@app.route('/predict',methods=['GET','POST'])
def predict():
    data=request.json.get('key')
    data=data.split(',')
    #print(data)
    calc=PredictDisease(data)
    output=calc.make_prediction()
    return jsonify(output)

@app.route('/cost',methods=['GET','POST'])
def cost():
    data=request.json.get('key')
    #data=data.split(',')
    print(data)
    calc=HealthExpenditure(data)
    output=calc.predict()
    output=round(output,2)
    x=output-0.5*output
    y=output+0.5*output
    string="("+str(x)+"-"+str(y)+")"
    print(output,string)
    return jsonify(string)
    #return jsonify("ssssssssssssssssss")

@app.route('/profile/<int:user_id>',methods=['GET'])
@token_required
def profile():
    data = request.data
    user = User.query.get(data['id'])
    if user:
        resp ={
             "id":user.id,
            'username':user.username,
            'email':user.email,
            "image_file":user.image_file,
            "posts":user.posts
        }
        return jsonify(resp)
    else:
        return jsonify({'message':'This is a protected'})
"""@app.route('/update/<int:post_id>',methods=['POST'])
@token_required
def update():
    data = request.data
    user = User.query.get(data['id'])
    user.
        return jsonify(resp)
    else:
        return jsonify({'message':'This is a protected'})"""

def getApp():
    return app


if __name__ == "__main__":
    app.run(debug=True,port=5000,host="0.0.0.0")
