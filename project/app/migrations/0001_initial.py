# Generated by Django 4.0 on 2023-01-20 07:50

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Users',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('firstName', models.CharField(max_length=255)),
                ('lastName', models.CharField(max_length=255)),
                ('dateOfBirth', models.DateField(max_length=8)),
                ('emailId', models.EmailField(max_length=100)),
                ('password', models.CharField(max_length=255)),
            ],
        ),
    ]
