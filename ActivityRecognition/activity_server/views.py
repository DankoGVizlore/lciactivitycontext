import json

from django.views.generic import View
from django.shortcuts import render_to_response
from django.template.context import RequestContext
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from activity_server.controler.store_data_record_controller import store_data_record
from activity_server.controler.fetch_data_record_controller import recognize_last_activity
from activity_server.models import activity_table


class HomeView(View):
    def get(self, request):
        return render_to_response("home.html", None, context_instance=RequestContext(request))


class RESTView(View):
    def post(self, request):
        try:
            store_data_record(json.loads(request.body))
        except ValueError, e:
            response = HttpResponse("{error:%s}" % e.message)
            request.status_code = 404
            return response

        response = HttpResponse('{}')
        response.status_code = 201
        return response

    def get(self, request):

        try:
            if 'uuid' not in request.GET:
                raise Exception("Bad request")

            if 'time' in request.GET:
                pass
            else:
                record = recognize_last_activity(request.GET['uuid'],
                                                 request.GET['alg'] if 'alg' in request.GET else 'svc',
                                                 request.GET['fg'] if 'fs' in request.GET else 'standard')

            response_text = "date_time:'%s', " % record.get("time")

            for i in xrange(len(record['vector'])):
                if i != len(record['vector']) - 1:
                    response_text += "%s:'%s'," % (activity_table.get(i+1), record['vector'][i])
                else:
                    response_text += "%s:'%s'}" % (activity_table.get(i+1), record['vector'][i])

            response = HttpResponse(response_text)
            response.status_code = 200
            return response
        except ValueError, e:
            response = HttpResponse("{error:%s}" % e.message)
            request.status_code = 404
            return response

    @csrf_exempt
    def dispatch(self, *args, **kwargs):
        return super(RESTView, self).dispatch(*args, **kwargs)