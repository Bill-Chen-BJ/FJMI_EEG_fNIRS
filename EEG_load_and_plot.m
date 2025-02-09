%% Before running this script, please ensure that the dataset was downloaded and saved in the "./Data/FineMI" directory
% create the folder for temporary files
clear all;
folder = "matlab_files";
if ~exist(folder, 'dir')
    mkdir(folder);
end
%% Note that the loading, preprocessing, and epoching of subject 1 is different from other subjects.
% subject1 load
fs = 250;

EEG = pop_loadcnt('./Data/FineMI/subject1/EEG/block1-4.cnt' , 'dataformat', 'int32');
EEG = eeg_checkset( EEG );
EEG = pop_resample( EEG, fs);
EEG = eeg_checkset( EEG );
data=EEG.data;
event=EEG.event;

data([33 43 65 66 67 68],:)=[];
filename=['matlab_files/subject1_block1-4','.mat']; 
save(filename,'data','event');
clear data event

EEG = pop_loadcnt('./Data/FineMI/subject1/EEG/block5.cnt' , 'dataformat', 'int32');
EEG = eeg_checkset( EEG );
EEG = pop_resample( EEG, fs);
EEG = eeg_checkset( EEG );
data=EEG.data;
event=EEG.event;
% data([1,2,5,6],:)=[];
data([33 43 65 66 67 68],:)=[];
filename=['matlab_files/subject1_block5','.mat']; 
save(filename,'data','event');
clear data event

EEG = pop_loadcnt('./Data/FineMI/subject1/EEG/block6.cnt' , 'dataformat', 'int32');
EEG = eeg_checkset( EEG );
EEG = pop_resample( EEG, fs);
EEG = eeg_checkset( EEG );
data=EEG.data;
event=EEG.event;

data([33 43 65 66 67 68],:)=[];
filename=['matlab_files/subject1_block6','.mat']; 
save(filename,'data','event');
clear data event

EEG = pop_loadcnt('./Data/FineMI/subject1/EEG/block7.cnt' , 'dataformat', 'int32');
EEG = eeg_checkset( EEG );
EEG = pop_resample( EEG, fs);
EEG = eeg_checkset( EEG );
data=EEG.data;
event=EEG.event;

data([33 43 65 66 67 68],:)=[];
filename=['matlab_files/subject1_block7','.mat']; 
save(filename,'data','event');
clear data event

EEG = pop_loadcnt('./Data/FineMI/subject1/EEG/block8.cnt' , 'dataformat', 'int32');
EEG = eeg_checkset( EEG );
EEG = pop_resample( EEG, fs);
EEG = eeg_checkset( EEG );
data=EEG.data;
event=EEG.event;

data([33 43 65 66 67 68],:)=[];
filename=['matlab_files/subject1_block8','.mat']; 
save(filename,'data','event');
clear data event

%% subject1 preprocessing: bandpass(4-40hz), CAR
load matlab_files/subject1_block1-4.mat
for i=1:160
    label1(i,1)=event(1,i).type;
end;
for i=1:160
    latency1(i,1)=event(1,i).latency;
end;

data=double(data);
lpass=4;
hpass=40;

filterorder = 3;
filtercutoff = [2*lpass/fs 2*hpass/fs];
[f_b, f_a] = butter(filterorder,filtercutoff);
for j = 1:62
    data(j,:) = filtfilt(f_b,f_a,data(j,:));
end

for i=1:62
    data1(i,:)=data(i,:)-mean(data,1);
end

clear data event
load matlab_files/subject1_block5.mat
for i=1:40
    label2(i,1)=event(1,i).type;
end;
for i=1:40
    latency2(i,1)=event(1,i).latency;
end;

data=double(data);
lpass=4;
hpass=40;

filterorder = 3;
filtercutoff = [2*lpass/fs 2*hpass/fs];
[f_b, f_a] = butter(filterorder,filtercutoff);
for j = 1:62;
    data(j,:) = filtfilt(f_b,f_a,data(j,:));
end

for i=1:62
    data2(i,:)=data(i,:)-mean(data,1);
end

clear data event
load matlab_files/subject1_block6.mat
for i=1:40
    label3(i,1)=event(1,i).type;
end;
for i=1:40
    latency3(i,1)=event(1,i).latency;
end;

data=double(data);
lpass=4;
hpass=40;

filterorder = 3;
filtercutoff = [2*lpass/fs 2*hpass/fs];
[f_b, f_a] = butter(filterorder,filtercutoff);
for j = 1:62;
    data(j,:) = filtfilt(f_b,f_a,data(j,:));
end

for i=1:62
    data3(i,:)=data(i,:)-mean(data,1);
end
clear data event
load matlab_files/subject1_block7.mat
for i=1:40
    label4(i,1)=event(1,i).type;
end;
for i=1:40
    latency4(i,1)=event(1,i).latency;
end;
data=double(data);
lpass=4;
hpass=40;

filterorder = 3;
filtercutoff = [2*lpass/fs 2*hpass/fs];
[f_b, f_a] = butter(filterorder,filtercutoff);
for j = 1:62;
    data(j,:) = filtfilt(f_b,f_a,data(j,:));
end

for i=1:62
    data4(i,:)=data(i,:)-mean(data,1);
end
clear data event
% Block 8
load matlab_files/subject1_block8.mat
for i=1:40
    label5(i,1)=event(1,i).type;
end;
for i=1:40
    latency5(i,1)=event(1,i).latency;
end;
data=double(data);
lpass=4;
hpass=40;

filterorder = 3;
filtercutoff = [2*lpass/fs 2*hpass/fs];
[f_b, f_a] = butter(filterorder,filtercutoff);
for j = 1:62;
    data(j,:) = filtfilt(f_b,f_a,data(j,:));
end

for i=1:62
    data5(i,:)=data(i,:)-mean(data,1);
end
% data1=single(data1);
clear data event

%% subject 1 epoching 
label=[label1; label2; label3; label4; label5];
tmin = -4; % start time of epoching (s)
t_window_length = 10; % trial length
win_start = tmin * fs; % start time of epoching (time step)
win_len = t_window_length * fs - 1;

for i=1:160
    data11(:,(1+(i-1)*fs*t_window_length):(i*fs*t_window_length))=data1(:,latency1(i)+win_start:latency1(i)+win_start++win_len);
end

for i=1:40
    data22(:,(1+(i-1)*fs*t_window_length):(i*fs*t_window_length))=data2(:,latency2(i)+win_start:latency2(i)+win_start++win_len);
end

for i=1:40
    data33(:,(1+(i-1)*fs*t_window_length):(i*fs*t_window_length))=data3(:,latency3(i)+win_start:latency3(i)+win_start++win_len);
end

for i=1:40
    data44(:,(1+(i-1)*fs*t_window_length):(i*fs*t_window_length))=data4(:,latency4(i)+win_start:latency4(i)+win_start++win_len);
end

for i=1:40
    data55(:,(1+(i-1)*fs*t_window_length):(i*fs*t_window_length))=data5(:,latency4(i)+win_start:latency4(i)+win_start++win_len);
end

data=[data11 data22 data33 data44 data55];
filename=['matlab_files/subject1','.mat']; 
save(filename,'data','label');
clear all;

%% subject 2-18 loading, preprocessing, and epoching
for subject_number =2:18
    n_blocks = 8;
    fs = 250;
    for i = 1:n_blocks
        EEG = pop_loadcnt("./Data/FineMI/subject"+subject_number+"/EEG/block"+i+".cnt", 'dataformat', 'int32');
        EEG = eeg_checkset( EEG );
        EEG = pop_resample( EEG, fs);
        EEG = eeg_checkset( EEG );
        data=EEG.data;
        event=EEG.event;
        data([33 43 65 66 67 68],:)=[];
        filename="matlab_files/subject"+subject_number+"_block"+i+".mat"; 
        save(filename,'data','event');
        clear data event
    end
    
    for num=1:8 % 8 blocks
        load("matlab_files/subject"+subject_number+"_block"+num+".mat")
        prefix_label = ['label', num2str(num)];
        prefix_latency = ['latency', num2str(num)];
        if num == 6 && subject_number == 5 % skip the first trial of block 6 in subject 5
            for i=2:41
                eval([prefix_label, '(i-1,1) = event(1,i).type;']);
                eval([prefix_latency, '(i-1,1) = event(1,i).latency;']);
            end
        else
            for i=1:40
                eval([prefix_label, '(i,1) = event(1,i).type;']);
                eval([prefix_latency, '(i,1) = event(1,i).latency;']);
            end
        end
        
        data=double(data);
        lpass=4;
        hpass=40;
        filterorder = 3;
        filtercutoff = [2*lpass/fs 2*hpass/fs];
        [f_b, f_a] = butter(filterorder,filtercutoff);
        for j = 1:62
            data(j,:) = filtfilt(f_b,f_a,data(j,:));
        end
        
        for i=1:62
            prefix_data = ['data', num2str(num)];
            eval([prefix_data, '(i,:)=data(i,:)-mean(data,1);'])
        end
        clear data event
    end
    

    
    tmin = -4; % start time of epoching
    t_window_length = 10; % trial length
    win_start = tmin * fs; % start time of epoching (time step)
    win_len = t_window_length * fs - 1;
    for i=1:40
        data11(:,(1+(i-1)*fs*t_window_length):(i*fs*t_window_length))=data1(:,latency1(i)+win_start:latency1(i)+win_start+win_len);
    end
    for i=1:40
        data22(:,(1+(i-1)*fs*t_window_length):(i*fs*t_window_length))=data2(:,latency2(i)+win_start:latency2(i)+win_start+win_len);
    end
    for i=1:40
        data33(:,(1+(i-1)*fs*t_window_length):(i*fs*t_window_length))=data3(:,latency3(i)+win_start:latency3(i)+win_start+win_len);
    end
    for i=1:40
        data44(:,(1+(i-1)*fs*t_window_length):(i*fs*t_window_length))=data4(:,latency4(i)+win_start:latency4(i)+win_start+win_len);
    end
    for i=1:40
        data55(:,(1+(i-1)*fs*t_window_length):(i*fs*t_window_length))=data5(:,latency5(i)+win_start:latency5(i)+win_start+win_len);
    end
    for i=1:40
        data66(:,(1+(i-1)*fs*t_window_length):(i*fs*t_window_length))=data6(:,latency6(i)+win_start:latency6(i)+win_start+win_len);
    end
    for i=1:40
        data77(:,(1+(i-1)*fs*t_window_length):(i*fs*t_window_length))=data7(:,latency7(i)+win_start:latency7(i)+win_start+win_len);
    end
    for i=1:40
        data88(:,(1+(i-1)*fs*t_window_length):(i*fs*t_window_length))=data8(:,latency8(i)+win_start:latency8(i)+win_start+win_len);
    end
    data=[data11 data22 data33 data44 data55 data66 data77 data88];
    label=[label1;label2;label3;label4;label5;label6;label7;label8];
    filename=["matlab_files/subject"+subject_number+".mat"]; 
    save(filename,'data','label');
    clear all;
end

%% compute ERSP of each subject
n_subjects = 18;
for number = 1:18
    % TF
    tic;
    n_classes = 8;
    load("matlab_files/subject"+number+".mat")
    for i=1:n_classes 
        MI_index{i}=find(label==i);
    end
    
    fs=250;%sampling rate
    trial_t=10;%trial length
    frames=fs*trial_t;%trial length (time step)
    tlimits=[-2000,8000];%time range
    cycles=0; %0 means using stft

    ERSP_mat = zeros(8,62,500,200);
    for i=1:62
        for j=1:8 
            index=MI_index{j};%
            for k=1:40
                temp_data(1,[1:frames]+frames*(k-1))=data(i,[1:frames]+frames*(index(k)-1));
            end;
            [ERSP,itc,powbase,times,freqs]=q_timef(temp_data,frames,tlimits,fs,cycles); 
            ERSP=ERSP(13:end,:);
            freqs=freqs(13:end);
            filename=['TF',num2str(i),'_',num2str(j),'.mat']; 
            folder = sprintf('matlab_files/tf_subject%s', number); % folder name
            fullpath = fullfile(folder, filename); % use fullfile to concate the filename and foleder name
            if ~exist(folder, 'dir') 
                mkdir(folder); 
            end
            save(fullpath, 'ERSP'); 
            ERSP_mat(j,i,:,:)=ERSP;
            clear temp_data ERSP;
        end
    end
    filename=["matlab_files/subject"+number+"_ersp.mat"]; 
    save(filename,'ERSP_mat','times','freqs');
    clear
end

%% load ERSP
n_subjects = 18;
ERSP_all_subjects=zeros(18,8,62,500,200);
subject_name_list=1:18;
for idx = 1:n_subjects
    number = subject_name_list(idx);
    tic;
    n_classes = 8;
    load("matlab_files/subject"+number+"_ersp.mat") 
    ERSP_all_subjects(idx,:,:,:,:) =  ERSP_mat;
end
%% Averaged ERSP plot 
fs=250;%sampling rate
trial_t=10;%trial length
frames=fs*trial_t;%trial length (time step)
tlimits=[-2000,8000];%time range
cycles=0; %0 means using stft

channel_names=["FP1";"FPZ";"FP2";"AF3";"AF4";"F7";"F5";"F3";"F1";"FZ";"F2";"F4";"F6";"F8";"FT7";"FC5";"FC3";"FC1";"FCZ";"FC2";"FC4";"FC6";"FT8";"T7";"C5";"C3";"C1";"CZ";"C2";"C4";"C6";"T8";"TP7";"CP5";"CP3";"CP1";"CPZ";"CP2";"CP4";"CP6";"TP8";"P7";"P5";"P3";"P1";"PZ";"P2";"P4";"P6";"P8";"PO7";"PO5";"PO3";"POZ";"PO4";"PO6";"PO8";"CB1";"O1";"OZ";"O2";"CB2"];
class_names=["Hand open/close";"Wrist flexion/extension";"Wrist abduction/adduction";"Elbow pronation/supination";"Elbow flexion/extension";"Shoulder pronation/supination";"Shoulder abduction/adduction";"Shoulder_flexion/extension";];
file_class_names=["Hand_open_close";"Wrist_flexion_extension";"Wrist_abduction_adduction";"Elbow_pronation_supination";"Elbow_flexion_extension";"Shoulder_pronation_supination";"Shoulder_abduction_adduction";"Shoulder_flexion_extension";];
channels_of_interest = [26];  % C3
ERSP_mean_cell={};
filename=['matlab_files/subjects_mean_ersp.mat']; 
if isfile(filename)
    load(filename)
end
for i=1:62
    for j=1:8 
        if isfile(filename)
            ERSP_mean = ERSP_mean_cell{j}{i};
        else
            ERSP_mean = squeeze(mean(ERSP_all_subjects(:,j,i,:,:),1)); % averaging across the trial dimension
            ERSP_mean_cell{j}{i}=ERSP_mean;
        end
        
        if ismember(i,channels_of_interest)
            pcolor(times/1000,freqs(1:132),ERSP_mean(1:132,1:200));  %5-35hz
            shading interp 
            
            xticks(0:2:6);
            xticklabels(["-2","0","2","4"]);
            
            hold on
            x=71;y=[1:1:103];
            
            plot(times(28)/1000*ones(1,66),freqs(1:2:132),'k--','LineWidth',2); %vertical dahsed line
            hold on;
            
            plot(times(75)/1000*ones(1,66),freqs(1:2:132),'k--','LineWidth',2); %vertical dahsed line
            hold on;
            
            plot(times(176)/1000*ones(1,66),freqs(1:2:132),'k--','LineWidth',2); %vertical dahsed line
            
            caxis([-3 1]);% scale
            colorbar('vert','fontsize',24,'fontweight','b');
            
            xlabel('times(s)','fontsize',24,'fontweight','b');
            ylabel('frequency(hz)','fontsize',24,'fontweight','b');
            title(channel_names(i)+": "+class_names(j),'fontsize',24);

            set(gca,'fontsize',24,'fontweight','b');
            exportgraphics(gcf,"matlab_files/tf_18subjects"+channel_names(i)+"_"+j+file_class_names(j)+".png","Resolution",600);
        end
    end
end
save(filename,'ERSP_mean_cell','times','freqs');

%% Topoplot
load matlab_files/subjects_mean_ersp.mat
alpha_ersp_cell={};
channel_names=["FP1";"FPZ";"FP2";"AF3";"AF4";"F7";"F5";"F3";"F1";"FZ";"F2";"F4";"F6";"F8";"FT7";"FC5";"FC3";"FC1";"FCZ";"FC2";"FC4";"FC6";"FT8";"T7";"C5";"C3";"C1";"CZ";"C2";"C4";"C6";"T8";"TP7";"CP5";"CP3";"CP1";"CPZ";"CP2";"CP4";"CP6";"TP8";"P7";"P5";"P3";"P1";"PZ";"P2";"P4";"P6";"P8";"PO7";"PO5";"PO3";"POZ";"PO4";"PO6";"PO8";"CB1";"O1";"OZ";"O2";"CB2"];
class_names=["Hand open/close";"Wrist flexion/extension";"Wrist abduction/adduction";"Elbow pronation/supination";"Elbow flexion/extension";"Shoulder pronation/supination";"Shoulder abduction/adduction";"Shoulder flexion/extension";];
file_class_names=["Hand_open_close";"Wrist_flexion_extension";"Wrist_abduction_adduction";"Elbow_pronation_supination";"Elbow_flexion_extension";"Shoulder_pronation_supination";"Shoulder_abduction_adduction";"Shoulder_flexion_extension";];
figure;
set(gcf,"Units","inches");
set(gcf,'position',[0,0,15,7]);
title("Alpha topoplot")
for i=1:8
    alpha_ersp=[];
    for j=1:62 
        mi_times_idx = [32:123]; % time range of MI performing
        alpha_freqs_idx =[21:42]; % frequency in alpha band
        alpha_ersp_channel = mean(ERSP_mean_cell{i}{j}(alpha_freqs_idx,mi_times_idx),"all");
        alpha_ersp = [alpha_ersp; alpha_ersp_channel];
    end
    subplot(2,4,i);
    title(class_names(i),'fontsize',24);
    title("",'fontsize',24);
    topoplot(alpha_ersp,'channel_location_60_neuroscan.locs'...
        ,'maplimits','maxmin','headrad',0.54,'hcolor',[0.5 0.5 0.5],'whitebk','on');% Topoplot requires the .locs file
    caxis([-1.5 0]);% scale
    colorbar('vert','fontsize',24,'fontweight','b');
    alpha_ersp_cell{i}=alpha_ersp;
end
exportgraphics(gcf,"matlab_files/tf_18subjects_alpha_topo.png","Resolution",600);

figure;
set(gcf,"Units","inches");
set(gcf,'position',[0,0,15,7]);
title("Beta topoplot")
beta_ersp_cell={};
for i=1:8
    beta_ersp=[];
    for j=1:62 
        mi_times_idx = [32:123]; % time range of MI performing
        beta_freqs_idx = [43:111]; % frequency in beta band
        beta_ersp_channel = mean(ERSP_mean_cell{i}{j}(beta_freqs_idx,mi_times_idx),'all');
        beta_ersp = [beta_ersp; beta_ersp_channel];
    end
    subplot(2,4,i);
    title(class_names(i),'fontsize',24);
    topoplot(beta_ersp,'channel_location_60_neuroscan.locs'...
        ,'maplimits','maxmin','headrad',0.54,'hcolor',[0.5 0.5 0.5],'whitebk','on');% Topoplot requires the .locs file
 
    caxis([-1.5 0]);% scale
    colorbar('vert','fontsize',24,'fontweight','b');
    beta_ersp_cell{i}=beta_ersp;
end
exportgraphics(gcf,"matlab_files/tf_18subjects_beta_topo.png","Resolution",600);  %Save the graph with DPI of 600
toc
clear










