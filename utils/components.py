def create_project_card(project):
    return f"""
    
    <div class='project-card'>
        <div class='support-badge'>SUPPORTED</div>
        <div class='favorite-icon'>⭐</div>
        <br>
        <h5 class='project-title'>{project['title']}</h5>
        <div class='metadata-item'>
            <span class='metadata-label'>Task:</span>
            <span class='metadata-value'>{project['task']}</span>
        </div>
        <div class='metadata-item'>
            <span class='metadata-label'>Model:</span>
            <span class='metadata-value'>{project['model']}</span>
        </div>
        <p class='project-description'>{project['description']}</p>
        <br>
        <a class="demo-button" color=white href="https://jocelynhsutjh.streamlit.app/{project['href']}"> Go to DEMO </a>
    </div>
    """

def create_project_section(projects):
    cards = "".join([create_project_card(project) for project in projects])
    return f"""
    <div class='title'><h3>PROJECT DEMO</h3></div>
    <br>
    <div class='card-container'>
        {cards}
    </div>
    """

def timeline_css():
    
    timeline_html = """
    <style>
    .timeline {
        position: relative;
        max-width: 1000px;
        margin: 0 auto;
        padding: 40px 0;
    }
    .timeline::after {
        content: '';
        position: absolute;
        width: 2px; /* Thinner line */
        background: linear-gradient(to bottom, grey, black 50%, grey); /* Gradient line */
        top: 0;
        bottom: 0;
        left: 50%;
        margin-left: -1px; /* Center the line */
        z-index: -1; /* Ensure the gradient line is behind everything */
    }
    .timeline-item {
        padding: 60px 0; /* Wider intervals */
        position: relative;
        width: 50%;
        opacity: 0;
        transform: translateY(20px);
        transition: opacity 0.6s ease-out, transform 0.6s ease-out;
    }
    .timeline-item:hover {
        background-color: #f0f0f0; /* Light grey background on hover */
        border-radius: 8px; /* Rounded corners on hover */
        transition: background-color 0.3s ease;
    }
    .timeline-item:nth-child(odd) {
        text-align: left;
    }
    .timeline-item:nth-child(even) {
        left: 50%;
        text-align: left;
        padding-left: 25px; /* Move text further to the right */
        
    }
    .timeline-item::after {
        content: '';
        position: absolute;
        width: 9px; /* Smaller dots */
        height: 9px;
        background-color: black; /* Black dots */
        border: 2px solid black; /* Black border for dots */
        top: 50%; /* Center vertically */
        transform: translateY(-50%);
        border-radius: 50%;
        z-index: 2; /* Dots in front of the line */
        transition: transform 0.3s ease, box-shadow 0.3s ease; /* Smooth transition for hover */
    }
    .timeline-item:hover::after {
        transform: translateY(-50%) scale(1.2); /* Enlarge dots on hover */
        box-shadow: 0 0 8px rgba(0, 0, 0, 0.5); /* Add shadow on hover */
    }
    .timeline-item:nth-child(odd)::after {
        right: -6px; /* Center dots on the line */
    }
    .timeline-item:nth-child(even)::after {
        left: -6px; /* Center dots on the line */
    }
    .timeline-item.visible {
        opacity: 1;
        transform: translateY(0);
    }
    .timeline-title {
        font-size: 1.5em;
        font-weight: bold;
        color: black;
        font-family: "Source Sans Pro", sans-serif;
    }
    .timeline-description {
        font-size: 1em;
        color: #555;
        margin-top: 10px;
        margin-right: 20px;
        font-family: "Source Sans Pro", sans-serif;
    }
    </style>
    <div class="timeline">
    """
    return timeline_html


def generate_timeline_html(data):
    
    timeline_html = timeline_css()

    for i, event in enumerate(data):
        timeline_html += f"""
        <div class="timeline-item" style="padding-bottom: 40px;">
            <div class="timeline-title">{event['title']}</div>
            <div class="timeline-description">{event['description']}</div>
        </div>
        """

    timeline_html += """
    </div>
    <script>
    document.addEventListener("DOMContentLoaded", function() {
        const timelineItems = document.querySelectorAll(".timeline-item");
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add("visible");
                }
            });
        }, { threshold: 0.5 });

        timelineItems.forEach(item => observer.observe(item));
    });
    </script>
    """
    return timeline_html