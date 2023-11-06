# Import the necessary libraries
import boto3 # for accessing Amazon mechanical Turk API[^1^][1]
import pandas as pd # for working with csv files and dataframes[^2^][2]
import random # for shuffling the images

# Create a connection to MTurk
mtc = boto3.client('mturk',
    aws_access_key_id = 'your_access_key_here',
    aws_secret_access_key = 'your_secret_key_here',
    region_name='us-east-1',
    endpoint_url = 'https://mturk-requester-sandbox.us-east-1.amazonaws.com' # use https://mturk-requester.us-east-1.amazonaws.com for production
)

# Load the csv file with image description and file names
df = pd.read_csv('images.csv')

# Load the csv file with survey items
survey = pd.read_csv('survey.csv')

# Define a function to create a HIT for each row of the dataframe
def create_hit(row):
    # Get the image description and file names from the row
    description = row['description']
    images = [row['image1'], row['image2'], row['image3'], row['image4']]
    
    # Shuffle the images to avoid order bias
    random.shuffle(images)
    
    # Create a HTML question using the description and images
    question = f"""
    <html>
    <head>
        <meta http-equiv='Content-Type' content='text/html; charset=UTF-8'/>
        <script type='text/javascript' src='https://s3.amazonaws.com/mturk-public/externalHIT_v1.js'></script>
    </head>
    <body>
        <form name='mturk_form' method='post' id='mturk_form' action='https://www.mturk.com/mturk/externalSubmit'>
        <input type='hidden' value='' name='assignmentId' id='assignmentId'/>
        <h1>Image Ranking Survey</h1>
        <p>Please read the following image description and rank the four images from best to worst according to each survey item.</p>
        <p><b>Image description:</b> {description}</p>
        <table border='1'>
            <tr>
                <td><img src='{images[0]}' width='200' height='200'/></td>
                <td><img src='{images[1]}' width='200' height='200'/></td>
                <td><img src='{images[2]}' width='200' height='200'/></td>
                <td><img src='{images[3]}' width='200' height='200'/></td>
            </tr>
            <tr>
                <td>{images[0]}</td>
                <td>{images[1]}</td>
                <td>{images[2]}</td>
                <td>{images[3]}</td>
            </tr>
        </table>
    """
    
    # Loop through the survey items and add a ranking question for each one
    for i, item in enumerate(survey['item']):
        question += f"""
        <p><b>Survey item {i+1}:</b> {item}</p>
        <p>Please rank the images from best to worst by dragging and dropping them into the boxes below.</p>
        <div id='ranking_{i+1}' style='display: flex; justify-content: space-between;'>
            <div style='width: 200px; height: 200px; border: 1px solid black; text-align: center; vertical-align: middle;'>Best</div>
            <div style='width: 200px; height: 200px; border: 1px solid black; text-align: center; vertical-align: middle;'>Second</div>
            <div style='width: 200px; height: 200px; border: 1px solid black; text-align: center; vertical-align: middle;'>Third</div>
            <div style='width: 200px; height: 200px; border: 1px solid black; text-align: center; vertical-align: middle;'>Worst</div>
        </div>
        <input type='hidden' name='ranking_{i+1}' id='ranking_{i+1}_value'/>
        <script type='text/javascript'>
            // Make the images draggable and droppable using jQuery UI
            $(function() {{
                $('img').draggable({{
                    revert: 'invalid',
                    helper: 'clone'
                }});
                $('#ranking_{i+1} div').droppable({{
                    accept: 'img',
                    drop: function(event, ui) {{
                        // Remove any previous image in the box
                        $(this).empty();
                        // Append the dropped image to the box
                        $(this).append(ui.draggable.clone());
                        // Update the hidden input value with the ranking order
                        var ranking = [];
                        $('#ranking_{i+1} img').each(function() {{
                            ranking.push($(this).attr('src'));
                        }});
                        $('#ranking_{i+1}_value').val(ranking.join(','));
                    }}
                }});
            }});
        </script>
        """
    
    # Close the HTML question and add some validation and submission code
    question += """
        <p><input type='button' id='submitButton' value='Submit' /></p>
        </form>
        <script language='Javascript'>turkSetAssignmentID();</script>
        <script type='text/javascript'>
            // Validate that all ranking questions are answered before submitting
            function validate() {
                var valid = true;
                for (var i = 1; i <= {len(survey)}; i++) {
                    var ranking = $('#ranking_' + i + '_value').val();
                    if (!ranking || ranking.split(',').length < 4) {
                        valid = false;
                        alert('Please rank all four images for survey item ' + i);
                        break;
                    }
                }
                return valid;
            }
            // Submit the form if validation passes
            $('#submitButton').click(function() {
                if (validate()) {
                    $('#mturk_form').submit();
                }
            });
        </script>
    </body>
    </html>
    """
    
    # Create a HIT using the HTML question
    response = mtc.create_hit(
        Title = 'Image Ranking Survey',
        Description = 'Rank four images from best to worst according to each survey item.',
        Keywords = 'image, ranking, survey',
        Reward = '0.00', # adjust the reward amount as needed
        MaxAssignments = 1, # adjust the number of assignments as needed
        LifetimeInSeconds = 3600, # adjust the HIT lifetime as needed
        AssignmentDurationInSeconds = 600, # adjust the assignment duration as needed
        AutoApprovalDelayInSeconds = 86400, # adjust the auto-approval delay as needed
        Question = question
    )
    
    # Return the HIT ID
    return response['HIT']['HITId']

# Apply the create_hit function to each row of the dataframe and store the HIT IDs in a new column
df['HITId'] = df.apply(create_hit, axis=1)

# Save the dataframe to a new csv file with the HIT IDs
df.to_csv('images_with_hit_ids.csv', index=False)
