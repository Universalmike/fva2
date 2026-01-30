import os
from flask import Flask, flash, render_template, redirect, request
from tasks import add

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', "super-secret")


from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS
from flask import send_file
from tasks import analyze_video_task, analyze_image_sequence_task

app = Flask(__name__)
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:8501", "http://localhost:3000", "*"],  # Allow all origins
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})
app.config['UPLOAD_FOLDER'] = '/tmp/uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok', 'message': 'API server is running'}), 200

@app.route('/api/analyze/video', methods=['POST'])
def analyze_video_endpoint():
    """Endpoint to submit video analysis job"""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file'}), 400
    
    file = request.files['video']
    mode = request.form.get('mode', 'standard')
    sampling_mode = request.form.get('sampling_mode', 'sampled')
    
    # Save file
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # Submit to queue
    task = analyze_video_task.delay(filepath, mode, sampling_mode)
    
    return jsonify({
        'job_id': task.id,
        'status': 'queued'
    })

@app.route('/api/analyze/images', methods=['POST'])
def analyze_images_endpoint():
    """Endpoint to submit image sequence analysis job"""
    if 'images' not in request.files:
        return jsonify({'error': 'No image files'}), 400
    
    files = request.files.getlist('images')
    sequence_type = request.form.get('sequence_type', 'ordered')
    mode = request.form.get('mode', 'standard')
    
    # Save files
    filepaths = []
    for file in files:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        filepaths.append(filepath)
    
    # Submit to queue
    task = analyze_image_sequence_task.delay(filepaths, sequence_type, mode)
    
    return jsonify({
        'job_id': task.id,
        'status': 'queued'
    })

@app.route('/api/job/<job_id>', methods=['GET'])
def get_job_status(job_id):
    """Check job status"""
    task = celery_app.AsyncResult(job_id)
    
    if task.state == 'PENDING':
        return jsonify({'status': 'pending'})
    elif task.state == 'SUCCESS':
        return jsonify({
            'status': 'completed',
            'result': task.result
        })
    elif task.state == 'FAILURE':
        return jsonify({
            'status': 'failed',
            'error': str(task.info)
        }), 500
    else:
        return jsonify({'status': task.state.lower()})

@app.route('/api/result/<job_id>', methods=['GET'])
def get_result(job_id):
    """Get analysis result"""
    result_path = f"/tmp/frame_analysis/{job_id}/result.json"
    
    if not os.path.exists(result_path):
        return jsonify({'error': 'Result not found'}), 404
    
    with open(result_path, 'r') as f:
        result = json.load(f)
    
    return jsonify(result)

from flask import send_file

@app.route('/api/export/pdf/<job_id>', methods=['GET'])
def export_pdf(job_id):
    """Download PDF report for a job"""
    try:
        # Find PDF file
        pdf_path = f"/tmp/frame_analysis/{job_id}/result_report.pdf"
        
        if not os.path.exists(pdf_path):
            # Try alternative naming (from PDF generator)
            result_json = f"/tmp/frame_analysis/{job_id}/result.json"
            if os.path.exists(result_json):
                # Generate PDF on-demand if not exists
                from pdfgneration import generate_pdf_report
                pdf_path = generate_pdf_report(result_json)
            else:
                return jsonify({'error': 'PDF not found'}), 404
        
        # Send file
        return send_file(
            pdf_path,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f'frame-analysis-{job_id}.pdf'
        )
    
    except Exception as e:
        logger.error(f"PDF export error: {e}")
        return jsonify({'error': str(e)}), 500

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    # Example usage
    print("Frame Analysis System - Backend")
    print("=" * 50)
    print("\nTo run the system:")
    print("\n1. Start Redis server:")
    print("   redis-server")
    print("\n2. Start Celery worker:")
    print("   celery -A frame_analysis worker --loglevel=info")
    print("\n3. Start Flask API:")
    print("   python frame_analysis.py")
    print("\n4. Submit jobs via API:")
    print("   curl -X POST -F 'video=@test.mp4' http://localhost:5000/api/analyze/video")
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
