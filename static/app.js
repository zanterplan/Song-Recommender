document.addEventListener('DOMContentLoaded', function () {
    // Song form (recommendation) variables
    const songForm = document.getElementById('song-form');
    const songNameInput = document.getElementById('song_name');
    const suggestionsDropdown = document.getElementById('suggestions-dropdown');

    // Filters variables
    const filtersButton = document.getElementById('filters-button');
    const filtersContainer = document.getElementById('filters-container');
    const genreInput = document.getElementById('genre-input');
    const genreDropdown = document.getElementById('genre-dropdown');

    // Upload audio (analyze) variables
    const uploadAudioForm = document.getElementById('upload-audio-form');
    const audioFileInput = document.getElementById('audio-file');

    // Upload audio form
    uploadAudioForm.addEventListener('submit', async function (event) {
        event.preventDefault();

        // Get the selected file
        const file = audioFileInput.files[0];
        if (!file) {
            alert('Please select an audio file to upload.');
            return;
        }
        
        // Remove suffix from the song
        suffix = ["mp3", "wav", "flac"]
        for (let i = 0; i < suffix.length; i++) {
            if (file.name.split(".")[1] == (suffix[i])) {
                break;
            }
            if (i == suffix.length - 1) {
                alert('File should be in one of these formats: .mp3, .wav, .flac.');
                return;
            }
        }

        // Create FormData to send the file
        const formData = new FormData();
        formData.append('file', file);
        formData.append('file_name', file.name);

        // Variables for properties
        var valence = -1;
        var loudness = -1;
        var energy = -1;
        var danceability = -1;
        var popularity = -1;        

        console.log(file.name);

        // Valence, loudness, energy, danceability
        try {
            // Send the file to the backend
            const response = await fetch('/upload-audio', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            // console.log(data);
            if (data !== undefined) {
                valence = data.valence;
                loudness = data.loudness;
                energy = data.energy;
                danceability = data.danceability;
                popularity = data.popularity;
                alert(`Song "${data.track_name}" by ${data.artists} with features:\n
                    happiness: ${valence.toFixed(3)}\n
                    loudness: ${loudness.toFixed(3)}\n
                    energy: ${energy.toFixed(3)}\n
                    danceability: ${danceability.toFixed(3)}\n
                    popularity: ${popularity.toFixed(3)}\n
                has been added to the database.`)
            } else {
                alert(data.error || 'An error occurred.');
            }
        } catch (error) {
            console.error('Error:', error);
            alert('Failed to upload and analyze the file.');
        }        

        /*console.log("valence: " + valence);
        console.log("loudness: " + loudness);
        console.log("energy: " + energy);
        console.log("danceability: " + danceability);
        console.log("popularity: " + popularity);*/
    });

    const sliders = [
        { id: "valence-slider", valueId: "valence-value" },
        { id: "popularity-slider", valueId: "popularity-value" },
        { id: "energy-slider", valueId: "energy-value" },
        { id: "danceability-slider", valueId: "danceability-value" },
        { id: "loudness-slider", valueId: "loudness-value" }
    ];

    // Form submission with "Get Recommendations" button
    songForm.addEventListener('submit', async function (event) {
        console.log(event);
        event.preventDefault();

        // Variables
        const songName = songNameInput.value;
        const topN = document.getElementById('top_n').value;
        const genre = genreInput.value;

        // Sliders
        const valence = document.getElementById(sliders[0].valueId).innerText;
        const popularity = document.getElementById(sliders[1].valueId).innerText;
        const energy = document.getElementById(sliders[2].valueId).innerText;
        const danceability = document.getElementById(sliders[3].valueId).innerText;
        const loudness = document.getElementById(sliders[4].valueId).innerText;

        // Recommendation method
        const recommendationMethod = document.querySelector('input[name="recommendation-method"]:checked').value;
        const randomMethod = document.querySelector('input[name="random-method"]').checked;

        /*console.log(valence);
        console.log(popularity);
        console.log(energy);
        console.log(danceability);
        console.log(loudness);*/

        if (songName.trim() === '') {
            alert('Please enter a song name');
            return;
        }

        try {
            // Send data to backend for recommendations
            const response = await fetch('/recommend', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ song_name: songName, top_n: topN, genre: genre, happiness: valence, 
                                       popularity: popularity, energy: energy, danceability: danceability,
                                       loudness: loudness, recommendation_method: recommendationMethod,
                                       random_method: randomMethod })
            });

            // Wait for recommendations
            const data = await response.json();
            if (data.recommendations) {
                displayRecommendations(data);
            } else {
                alert(data.error || 'An error occurred.');
            }
        } catch (error) {
            console.error('Error:', error);
            alert('Failed to fetch recommendations.');
        }
    });

    // Function to display recommendations
    function displayRecommendations(data) {
        const recommendationsList = document.getElementById('recommendations-list');
        const recommendationInfo = document.getElementById('recommendation-info');
        recommendationsList.innerHTML = '';

        const identifiedSong = data.identified_song;
        recommendationInfo.textContent = `Recommendations for "${identifiedSong.name}" by "${identifiedSong.artist}"`;
        
        // Display each recommendation as a card
        data.recommendations.forEach((rec) => {
            const card = document.createElement('div');
            card.className = 'song-card';
            
            // Display image
            const img = document.createElement('img');
            img.src = rec.image_url || 'static/default_cover.png';
            img.alt = `${rec.title} cover`;

            // Display title and artist
            const title = document.createElement('h3');
            title.textContent = rec.title;

            const artist = document.createElement('p');
            artist.textContent = rec.artist;

            // Create and append card
            card.appendChild(img);
            card.appendChild(title);
            card.appendChild(artist);

            recommendationsList.appendChild(card);
        });
    }

    // Typing in the song name input field
    songNameInput.addEventListener('input', async function () {
        const songName = songNameInput.value.trim();

        if (songName === '') {
            suggestionsDropdown.style.display = 'none';
            return;
        }

        // Call the API to get song suggestions
        try {
            const response = await fetch(`/search_suggestions?query=${songName}`);
            const data = await response.json();
            
            if (data.suggestions && data.suggestions.length > 0) {
                showSuggestions(data.suggestions);
            } else {
                suggestionsDropdown.style.display = 'none';
            }
        } catch (error) {
            console.error('Error:', error);
            suggestionsDropdown.style.display = 'none';
        }
    });

    // Show suggestions in the dropdown
    function showSuggestions(suggestions) {
        suggestionsDropdown.innerHTML = '';

        suggestions.slice(0, 10).forEach(function (suggestion) {
            // Get list properties
            const listItem = document.createElement('li');
            listItem.textContent = `${suggestion.track_name} - ${suggestion.artists}`;
            listItem.dataset.songName = suggestion.track_name;

            // console.log(suggestion);

            // Display clicked suggestion
            listItem.addEventListener('click', function () {
                songNameInput.value = suggestion.track_name + " - " + suggestion.artists;
                suggestionsDropdown.style.display = 'none';
            });

            // If clicked elsewhere
            addEventListener('click', function () {
                suggestionsDropdown.style.display = 'none';
            });

            // Append suggestion to dropdown
            suggestionsDropdown.appendChild(listItem);
        });

        suggestionsDropdown.style.display = 'block';
    }

    // Toggle the filters container visibility
    filtersButton.addEventListener('click', function () {
        if (filtersContainer.style.display === 'none' || !filtersContainer.style.display) {
            // Display filters
            filtersContainer.style.display = 'block';
            filtersButton.textContent = "Filters ▲";
            document.getElementById('filter-info').style.display = 'block';
        } else {
            // Hide filters
            filtersContainer.style.display = 'none';
            filtersButton.textContent = "Filters ▼";
            document.getElementById('filter-info').style.display = 'none';
        }
    });

    // Genre selection
    genreInput.addEventListener('input', async function () {
        const query = genreInput.value.trim();

        if (query.length > 0) {
            const response = await fetch(`/get_genres?query=${query}`);
            const genres = await response.json();

            // Clear and populate the dropdown
            genreDropdown.innerHTML = '';
            genres.slice(0, 10).forEach(genre => {
                const li = document.createElement('li');
                li.textContent = genre;
                genreDropdown.appendChild(li);

                li.addEventListener('click', () => {
                    genreInput.value = genre;
                    genreDropdown.style.display = 'none';
                });
            });

            genreDropdown.style.display = 'block';
        } else {
            genreDropdown.style.display = 'none';
        }
    });

    // Hide dropdown when clicking outside
    document.addEventListener('click', function (e) {
        if (!genreInput.contains(e.target) && !genreDropdown.contains(e.target)) {
            genreDropdown.style.display = 'none';
        }
    });

    // Update sliders
    sliders.forEach(({ id, valueId }) => {
        const slider = document.getElementById(id);
        const valueDisplay = document.getElementById(valueId);

        slider.addEventListener("input", () => {
            valueDisplay.textContent = slider.value;
        });
    });    
});
