

$(function () {
            $("#submit").click(function (event) {
                var box = document.getElementById('box');
                var context = box.getContext('2d');
                var imgData = context.getImageData(0, 0, box.width, box.height).data;
                $.post('get_result', {"img_data": imgData.toLocaleString()}, function (json, textStatus) {
                    $("#result").text(json["status"])
                });
            })
});