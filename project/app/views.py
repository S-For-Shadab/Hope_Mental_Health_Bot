from http.client import HTTPResponse
from django.shortcuts import render
from .models import Users
from .classOfUser import classOfUser
from django.shortcuts import redirect
dict={}
# Create your views here.

def index(request):
    return render(request, 'chatpage.html')


def loginpage(request):
    if request.method == "POST":
        emailId = request.POST.get('emailId')
        password = request.POST.get('password')
        usersinwebsite = Users.objects.all()

        if emailId in dict.keys():
            print("present")
            # return render(request,'chatpage.html',{'id':id,'name':name})4
            print(dict)

        else:
            print("not present")
            print(dict.keys())

        for user in usersinwebsite:
            if user.emailId == emailId and user.password == password:
                name=user.firstName
                id=(int)(user.id)
                return render(request,'chatpage.html',{'id':id,'name':name})

    return render(request, 'loginpage.html')


def homepage(request,id):
    if request.method == "GET":
        return render(request,'homepage.html')


    

def registerpage(request):

    
    return render(request,'register.html')
 
def addEntryOfUser(request):
    if request.method == "POST":
        emailId = request.POST.get('email')
        password = request.POST.get('password')
        fname=request.POST.get('firstname')
        lname=request.POST.get('lastname')
        dob=request.POST.get('dob') 
        entry=Users(firstName=fname,lastName=lname,dateOfBirth=dob,emailId=emailId,password=password)
        newUser=classOfUser(fname,lname,dob,emailId,password) 
        entry.save()
        dict.update({emailId:newUser}) 

        return redirect('/loginpage/')
