pipeline {
    agent any
    
    stages {
        stage('Packaging') {
            steps {
                sh 'docker build --pull --rm -f Dockerfile -t loan-prediction-app:latest .'
            }
        }

        stage('Push to DockerHub') {
            steps {
                withDockerRegistry(credentialsId: 'dockerhub', url: 'https://index.docker.io/v1/') {
                    sh 'docker tag loan-prediction-app:latest tuanhuu3264/hackathon:latest'
                    sh 'docker push tuanhuu3264/hackathon:latest'
                }
            }
        }

        stage('Deploy to Production') {
            steps {
                echo 'Deploying and cleaning'
                sh 'docker container stop loan-prediction-app || echo "this container does not exist"'
                sh 'echo y | docker system prune'
                sh 'docker container run -d --name loan-prediction-app -p 8000:8000 tuanhuu3264/hackathon'
            }
        }
    }
    
    post {
        always {
            cleanWs()
        }
    }
} 