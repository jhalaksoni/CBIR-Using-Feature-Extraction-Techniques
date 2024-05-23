from django.contrib import messages

def message_context(request):
    return {
        'messages': messages.get_messages(request)
    }