/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 *
 *  Modified on: May 2017
 *      Author: Tianzi Harrison
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
    //   x, y, theta and their uncertainties from GPS) and all weights to 1.
    // Add random Gaussian noise to each particle.
    // NOTE: Consult particle_filter.h for more information about this method (and others in this file).

    num_particles = 100;
    particles.resize(num_particles);
    weights.resize(num_particles);

    default_random_engine gen;

    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);

    double initial_weight = 1.0/num_particles; // so that weights sum to 1

    for (int i = 0; i < num_particles; ++i) {
        Particle &p = particles[i];

        p.id = i;
        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.theta = dist_theta(gen);
        p.weight = initial_weight;

        weights[i] = initial_weight;
    }

    is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    // TODO: Add measurements to each particle and add random Gaussian noise.
    // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
    //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
    //  http://www.cplusplus.com/reference/random/default_random_engine/
    default_random_engine gen;

    normal_distribution<double> dist_x(0.0, std_pos[0]);
    normal_distribution<double> dist_y(0.0, std_pos[1]);
    normal_distribution<double> dist_theta(0.0, std_pos[2]);

    for(int i=0; i<num_particles; i++){
        Particle &p = particles[i];

        if(fabs(yaw_rate)<0.0001){
            p.x += velocity * delta_t * cos(p.theta);
            p.y += velocity * delta_t * sin(p.theta);

        }else{
            p.x += (velocity/yaw_rate) * (sin(p.theta + yaw_rate*delta_t) - sin(p.theta));
            p.y += (velocity/yaw_rate) * (cos(p.theta) - cos(p.theta+yaw_rate * delta_t));
            p.theta += yaw_rate * delta_t;
        }

        p.x += dist_x(gen);
        p.y += dist_y(gen);
        p.theta += dist_theta(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
    // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
    //   observed measurement to this particular landmark.
    // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
    //   implement this method and use it as a helper during the updateWeights phase.

    vector<LandmarkObs> closest_landmarks(observations.size());

    default_random_engine gen;
    bernoulli_distribution distribution(0.5);

    for(int i = 0; i < observations.size(); i++){
        LandmarkObs best;
        double min_distance = 0.0;

        for(int j = 0; j < predicted.size(); j++){
            double tmp_distance = dist(observations[i].x, observations[i].y, predicted[i].x, predicted[j].y);

            // Flips a coin to decide whether to associate a predicted landmark to an observation point
            // Without this step, if there are multiple predicted landmarks that have the same distance to
            // an observation point, the for loop will always force the predicted landmark with the highest
            // index to be associated with the observation point.

            if(tmp_distance < min_distance || (tmp_distance == min_distance && distribution(gen))){
                min_distance = tmp_distance;
                best = predicted[j];
            }
        }
        closest_landmarks[i] = best;
    }
    observations = closest_landmarks;

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   std::vector<LandmarkObs> observations, Map map_landmarks) {
    // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
    //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
    //   according to the MAP'S coordinate system. You will need to transform between the two systems.
    //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
    //   The following is a good resource for the theory:
    //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
    //   and the following is a good resource for the actual equation to implement (look at equation
    //   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account
    //   for the fact that the map's y-axis actually points downwards.)
    //   http://planning.cs.uiuc.edu/node99.html

    double sum_weights = 0.0;

    for(int i = 0; i < num_particles; i++){

        double weight = 1.0;

        for (int j = 0; j < observations.size(); j++){

            // Step 1: transforms observed landmarks from the car's coordinates to the map's coordinates,
            // in respect to particle[i].

            LandmarkObs obs_landmark;

            obs_landmark.id = observations[j].id;
            obs_landmark.x = particles[i].x + observations[j].x * cos(particles[i].theta) -
                             observations[j].y * sin(particles[i].theta);
            obs_landmark.y = particles[i].y + observations[j].x * sin(particles[i].theta) +
                             observations[j].y * cos(particles[i].theta);

            // Step 2: associate the transformed landmarks with map landmarks.

            default_random_engine gen;
            bernoulli_distribution distribution(0.5);

            Map::single_landmark_s closest_landmark = map_landmarks.landmark_list[0];

            double min_distance = dist(obs_landmark.x, obs_landmark.y, closest_landmark.x_f, closest_landmark.y_f);

            for(size_t j = 1; j<map_landmarks.landmark_list.size(); j++){

                Map::single_landmark_s current_landmark = map_landmarks.landmark_list[j];

                double cur_distance = dist(obs_landmark.x, obs_landmark.y, current_landmark.x_f, current_landmark.y_f);

                if(cur_distance < min_distance || (cur_distance == min_distance && distribution(gen))){

                    closest_landmark = map_landmarks.landmark_list[j];
                    min_distance = cur_distance;
                }
            }

            // Step 3: calculate the weight of a single observed landmark in respect to the closest landmark if
            // particle[i] is where the car is.

            weight *= 1 / (2.0 * M_PI * std_landmark[0] * std_landmark[1]) *
                      exp(-0.5 * (pow((closest_landmark.x_f - obs_landmark.x), 2) / pow(std_landmark[0], 2) +
                                  pow((closest_landmark.y_f - obs_landmark.y), 2) / pow(std_landmark[1], 2)));

        }

        // Step 4: multiply all the calculated weights togehter to get the final weight. 
        sum_weights += weight;
        particles[i].weight = weight;
    }

    // update weights and particles.
    for(int i = 0; i < num_particles; i++){
        particles[i].weight /= sum_weights;
        weights[i] = particles[i].weight;
    }
}

void ParticleFilter::resample() {
    // TODO: Resample particles with replacement with probability proportional to their weight.
    // NOTE: You may find std::discrete_distribution helpful here.
    //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    std::vector<Particle> particles_old(particles);
    default_random_engine gen;

    // std::size_t is the unsigned integer type of the result of the sizeof operator
    discrete_distribution<std::size_t> d(weights.begin(), weights.end());

    for(Particle& p: particles){
        p = particles_old[d(gen)];
    }

}

//Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x,
//                                         std::vector<double> sense_y)
//{
//	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
//	// associations: The landmark id that goes along with each listed association
//	// sense_x: the associations x mapping already converted to world coordinates
//	// sense_y: the associations y mapping already converted to world coordinates
//
//	//Clear the previous associations
//	particle.associations.clear();
//	particle.sense_x.clear();
//	particle.sense_y.clear();
//
//	particle.associations= associations;
// 	particle.sense_x = sense_x;
// 	particle.sense_y = sense_y;
//
// 	return particle;
//}

//string ParticleFilter::getAssociations(Particle best)
//{
//	vector<int> v = best.associations;
//	stringstream ss;
//    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
//    string s = ss.str();
//    s = s.substr(0, s.length()-1);  // get rid of the trailing space
//    return s;
//}
//string ParticleFilter::getSenseX(Particle best)
//{
//	vector<double> v = best.sense_x;
//	stringstream ss;
//    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
//    string s = ss.str();
//    s = s.substr(0, s.length()-1);  // get rid of the trailing space
//    return s;
//}
//string ParticleFilter::getSenseY(Particle best)
//{
//	vector<double> v = best.sense_y;
//	stringstream ss;
//    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
//    string s = ss.str();
//    s = s.substr(0, s.length()-1);  // get rid of the trailing space
//    return s;
//}

void ParticleFilter::write(std::string filename) {
    // You don't need to modify this file.
    std::ofstream dataFile;
    dataFile.open(filename, std::ios::app);
    for (int i = 0; i < num_particles; ++i) {
        dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
    }
    dataFile.close();
}