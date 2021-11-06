from django.db import models

# Create your models here.

class NodeLog(models.Model):

    #node_lan_address is the LAN IP of the node
    node_lan_address = models.CharField(max_length=16, null=False)

    #mode can either be 'orchestrator' or 'computational'
    mode  = models.CharField(max_length=15, null=False)

    #creation_date is the creation date of when the instance is inserted to DB
    creation_date = models.DateTimeField(auto_now_add=True, null=False)

    #is_active means whether the node is on or not
    is_active = models.BooleanField(default=False, null=False)

    #is_working  means that this node is active and computing stuff
    is_working = models.BooleanField(default=False, null=False)
