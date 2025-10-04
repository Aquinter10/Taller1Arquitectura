$(document).on('submit','#post-form',function(e){
    e.preventDefault();

    $.ajax({
      type:'POST',
      url:'/send',
      data:{
          username:$('#username').val(),
          chat_id:$('#chat_id').val(),
          message:$('#message').val(),
        csrfmiddlewaretoken:$('input[name=csrfmiddlewaretoken]').val(),
      },
      success: function(data){
      }
    });
    document.getElementById('message').value = ''
  });