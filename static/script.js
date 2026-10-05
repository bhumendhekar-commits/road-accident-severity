document
    .getElementById("predictionForm")
    .addEventListener("submit", async function (event) {

        event.preventDefault();

        const button = document.querySelector(".predict-btn");

        button.innerHTML = "⏳ Analyzing Accident...";
        button.disabled = true;

        const data = {

            Age_band_of_driver:
                document.getElementById("Age_band_of_driver").value,

            Sex_of_driver:
                document.getElementById("Sex_of_driver").value,

            Driving_experience:
                document.getElementById("Driving_experience").value,

            Type_of_vehicle:
                document.getElementById("Type_of_vehicle").value,

            Area_accident_occured:
                document.getElementById("Area_accident_occured").value,

            Road_surface_type:
                document.getElementById("Road_surface_type").value,

            Road_surface_conditions:
                document.getElementById("Road_surface_conditions").value,

            Light_conditions:
                document.getElementById("Light_conditions").value,

            Weather_conditions:
                document.getElementById("Weather_conditions").value,

            Type_of_collision:
                document.getElementById("Type_of_collision").value,

            Number_of_vehicles_involved:
                document.getElementById("Number_of_vehicles_involved").value,

            Number_of_casualties:
                document.getElementById("Number_of_casualties").value,

            Cause_of_accident:
                document.getElementById("Cause_of_accident").value
        };


        try {

            const response = await fetch("/predict", {

                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify(data)

            });


            const result = await response.json();


            if (result.success) {

                document.getElementById("predictionText").innerText =
                    "Prediction: " + result.prediction;


                const minor =
                    result.probabilities["Minor"] || 0;

                const serious =
                    result.probabilities["Serious"] || 0;

                const severe =
                    result.probabilities["Severe"] || 0;


                document.getElementById("minorProbability").innerText =
                    minor + "%";

                document.getElementById("seriousProbability").innerText =
                    serious + "%";

                document.getElementById("severeProbability").innerText =
                    severe + "%";


                setTimeout(() => {

                    document.getElementById("minorBar").style.width =
                        minor + "%";

                    document.getElementById("seriousBar").style.width =
                        serious + "%";

                    document.getElementById("severeBar").style.width =
                        severe + "%";

                }, 100);


                document.getElementById("result").style.display =
                    "block";


                document.getElementById("result")
                    .scrollIntoView({
                        behavior: "smooth",
                        block: "center"
                    });

            } else {

                alert("Prediction Error: " + result.error);

            }

        }

        catch (error) {

            console.error(error);

            alert(
                "Server error. Please make sure Flask is running."
            );

        }

        finally {

            button.innerHTML =
                '<span>🔮</span> Predict Accident Severity <span class="arrow">→</span>';

            button.disabled = false;

        }

    });