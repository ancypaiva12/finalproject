# finalproject  

## changes made  
### line 162-165  
```python 			
ret, buffer = cv2.imencode('.jpg', frame)
frame = buffer.tobytes()
yield (b'--frame\r\n'
    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
```  
cv2.imencode('.jpg', frame) encodes the video frame into a JPEG image format, and buffer stores array of bytes representing the encoded image.  
frame = buffer.tobytes() converts buffer into a  python like bytes object, which can be sent over web.  
Finally yields a response with header as image/jpeg.  

### line 169-177
```python
@app.route('/')
def home():
    return render_template("home.html")
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
```
home route(127.0.0.1:5000/) loads home.html file from templates, which further extracts video from route video_feed  
css can be made as per need on home.html  

### commented lines
line 20, 143, 181 commented to temporary remove temperature feature, uncomment to add it back.  

### PS  
ommitted threading, haven't seen it being any effective here  
                