from django.db import models 
 
class Users(models.Model):
  # id = models.BigAutoField(primary_key=True)
  firstName = models.CharField(max_length=255)
  lastName = models.CharField(max_length=255)
  dateOfBirth=models.DateField(max_length=8)
  emailId=models.EmailField(max_length=100)
  password=models.CharField(max_length=255)

  def __str__(self):
    return self.emailId






   


